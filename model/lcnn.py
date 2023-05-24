"""
This code is modified version of LCNN baseline
from ASVSpoof2021 challenge - https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py
"""
import sys
import torch.nn as nn
import torch
import torch.nn as torch_nn
import numpy as np

try:
    from model import frontends
except:
    import inspect
    import os

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import frontends

    # TODO(PK): current implementation works only on CUDA


# For blstm
class BLSTMLayer(torch_nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch_nn.LSTM(
            input_dim,
            output_dim // 2,
            bidirectional=True
        )

    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)


class MaxFeatureMap2D(torch_nn.Module):
    """ Max feature map (along 2D)

    MaxFeatureMap2D(max_dim=1)

    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)


    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)

    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)

    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


##############
## FOR MODEL
##############

class BaseLCNN(torch_nn.Module):
    """ Model definition
    """

    def __init__(self, **kwargs):
        super().__init__()
        input_channels = 2
        num_coefficients = 60

        # Working sampling rate
        self.num_coefficients = num_coefficients

        # dimension of embedding vectors
        # here, the embedding is just the activation before sigmoid()
        self.v_emd_dim = 2

        # it can handle models with multiple front-end configuration
        # by default, only a single front-end

        self.m_transform = torch_nn.Sequential(
            torch_nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),

            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.BatchNorm2d(48, affine=False),

            torch_nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(48, affine=False),
            torch_nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),

            torch.nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(64, affine=False),
            torch_nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),

            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch_nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Dropout(0.7)
        )

        self.m_before_pooling = torch_nn.Sequential(
            BLSTMLayer((self.num_coefficients // 16) * 32, (self.num_coefficients // 16) * 32),
            BLSTMLayer((self.num_coefficients // 16) * 32, (self.num_coefficients // 16) * 32)
        )

        self.m_output_act = torch_nn.Linear((self.num_coefficients // 16) * 32, self.v_emd_dim)

    def _compute_embedding(self, x):
        """ definition of forward method
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        # x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)

        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        output_emb = torch.zeros(
            [batch_size, self.v_emd_dim],
            device=x.device,
            dtype=x.dtype
        )

        # compute scores for each sub-models
        idx = 0

        # compute scores
        #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
        #  2. compute hidden features
        x = x.permute(0, 1, 3, 2)
        # print(x.shape)
        hidden_features = self.m_transform(x)
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        # print(hidden_features.shape)
        frame_num = hidden_features.shape[1]

        hidden_features = hidden_features.view(batch_size, frame_num, -1)
        #  4. pooling
        #  4. pass through LSTM then summingc
        hidden_features_lstm = self.m_before_pooling(hidden_features)

        #  5. pass through the output layer
        hidden_features2 = (hidden_features_lstm + hidden_features).mean(1)
        # print((hidden_features_lstm + hidden_features).mean(1).shape)
        tmp_emb = self.m_output_act((hidden_features_lstm + hidden_features).mean(1))
        output_emb[idx * batch_size: (idx + 1) * batch_size] = tmp_emb

        return output_emb, hidden_features2

    def _compute_score(self, feature_vec):
        # feature_vec is [batch * submodel, 1]
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        classifier_out, feature = self._compute_embedding(x)

        return classifier_out, feature


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=2, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x1 = channel_out * x
        x_out = x + x1
        return x_out


class LCNN(BaseLCNN):
    """ Model definition
    """

    def __init__(self, device: str = "cuda", frontend_algorithm=["mfcc"], **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.ne = torch_nn.Sequential(
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(2, 2, 1, 1, 0, 1, 1, bias=False))
        self.ca = CBAMLayer(2)

        frontend_name = frontend_algorithm
        self.frontend = frontends.get_frontend(frontend_name)
        print(f"Using {frontend_name} frontend")

    def _compute_frontend(self, x):
        frontend = self.frontend(x)
        # print(frontend.shape)
        if frontend.ndim < 4:
            return frontend.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
        return frontend  # (bs, n, n_lfcc, frames)

    def feature(self, x):
        x1 = self.ne(x)
        x = x + x1
        x = self.ca(x)
        return x

    def forward(self, x):
        # print(x.shape)
        # x = self._compute_frontend(x)
        # print(x.shape)
        classifier_out, feature = self._compute_embedding(x)
        # print(feature.shape)

        return classifier_out, feature


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 800  # be same to the max_iter of config.py

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(96, 96)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(96, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.grl_layer = GRL()

    def forward(self, feature):
        feature = self.grl_layer.forward(feature)
        feature = self.fc1(feature)
        feature = self.relu(feature)
        feature = self.dropout(feature)
        feature = self.fc2(feature)
        return feature


# def init_weights(m):
#   classname = m.__class__.__name__
#   if classname.find('SeparableConv2d') != -1:
#     m.c.weight.data.normal_(0.0, 0.01)
#     if m.c.bias is not None:
#       m.c.bias.data.fill_(0)
#     m.pointwise.weight.data.normal_(0.0, 0.01)
#     if m.pointwise.bias is not None:
#       m.pointwise.bias.data.fill_(0)
#   elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
#     m.weight.data.normal_(0.0, 0.01)
#     if m.bias is not None:
#       m.bias.data.fill_(0)
#   elif classname.find('BatchNorm') != -1:
#     m.weight.data.normal_(1.0, 0.01)
#     m.bias.data.fill_(0)
#   elif classname.find('LSTM') != -1:
#     for i in m._parameters:
#       if i.__class__.__name__.find('weight') != -1:
#         i.data.normal_(0.0, 0.01)
#       elif i.__class__.__name__.find('bias') != -1:
#         i.bias.data.fill_(0)
# # dg_model = DG_model(model = 'xception')
# # dg_model.apply(init_weights)

class DGModel:
    def __init__(self, frontend_algorithm="mfcc"):
        dg_model = LCNN(frontend_algorithm)
        # if load_pretrain:
        #   state_dict = torch.load('/home/lifan/PycharmProjects/PEL-SSDG-CVPR2020/pretrained_model/xception-b5690688.pth')
        #   # state_dict = torch.load('/home/lifan/PycharmProjects/PEL-SSDG-CVPR2020/experiment_pg_star_glow_style/pg_star_style_to_glow/glow_checkpoint/xception/best_model/model_best_0.02501_48.pth.tar')
        #   for name, weights in state_dict.items():
        #     if 'pointwise' in name:
        #       state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        #   state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        #   dg_model.load_state_dict(state_dict, False)
        # else:
        #   dg_model.apply(init_weights)
        self.dg_model = dg_model
# #

# if __name__ == "__main__":
#
#     device = "cuda"
#     # print("Definition of model")
#     # model = LCNN(input_channels=1, num_coefficients=80)
#     # batch_size = 12
#     # mock_input = torch.rand((batch_size, 1, 80, 404))
#     # output = model(mock_input)
#     # print(output.shape)
#
#
#     print("Definition of model")
#     model = LCNN(input_channels=2, num_coefficients=80, device=device, frontend_algorithm=["mel_spec"])
#     model = model.to(device)
#     batch_size = 12
#     mock_input = torch.rand((batch_size, 64_600,), device=device)
#     output = model(mock_input)
#     print(output.shape)
