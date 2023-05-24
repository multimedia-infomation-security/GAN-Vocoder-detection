class DefaultConfigs(object):
    seed = 666
    weight_decay = 0.0005
    momentum = 0.9
    # learning rate
    init_lr =  0.1
    lr_epoch_1 = 0
    lr_epoch_2 = 50
    # model
    pretrained = False
    model = 'xception'
    # training parameters
    gpus = "0"
    batch_size = 4
    norm_flag = True
    max_iter = 25000
    lambda_triplet = 2
    lambda_adreal = 0.5
    #
    # tgt_best_model_name = 'model_best_0.09635_28.pth.tar'###mel_large
    # tgt_best_model_name = 'model_best_0.00289_30.pth.tar'###parallel_wave
    # tgt_best_model_name = 'model_best_0.00367_4.pth.tar'###stylemel
    # tgt_best_model_name = 'model_best_0.004_13.pth.tar'###hifi

    # source data information
    #stylemel / hifi / parallel_wave / mel_large
    src1_data = 'mel_large'
    src1_train_num_frames = 1
    src2_data = 'parallel_wave'
    src2_train_num_frames = 1
    src3_data = 'hifi'#multi_mel
    src3_train_num_frames = 1
    # target data information
    tgt_data = 'stylemel'
    tgt_test_num_frames = 1
    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
