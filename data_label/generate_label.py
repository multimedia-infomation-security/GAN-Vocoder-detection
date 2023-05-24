import os
import json
import sys
import glob

# change your data path
data_dir = '/home/lifan/project_audio/project-NN-Pytorch-scripts-master/project/SSGD_audio/DATA/'
def audio_process():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './parallel_wave/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    # f_train = open(label_save_dir + 'train_label.json', 'w')
    # f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'kss_parallel_wave/'
    fprint_path=data_dir+'wavegan_fprint/'
    # print(dataset_path)
    path_list = glob.glob(dataset_path + '*.wav', recursive=True)
    # fprint_path_list =glob.glob(fprint_path + '*.npy', recursive=True)
    # print(path_list)
    path_list.sort()
    for i in range(len(path_list)):
        file_name=path_list[i].split('/')[-1].split('.')[0]
        flag = path_list[i].find('_gen')
        # print(flag)
        if(flag != -1):
            label = 0
        else:
            label = 1
        dict = {}
        dict['audio_path'] = path_list[i]
        dict['photo_label'] = label
        dict['fprint_path']=fprint_path+file_name+".npy"
        # flag = path_list[i].find('/train/')
        # if (flag != -1):
        #     train_final_json.append(dict)
        # else:
        #     test_final_json.append(dict)
        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    # json.dump(train_final_json, f_train, indent=4)
    # f_train.close()
    # json.dump(test_final_json, f_test, indent=4)
    # f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()



if __name__=="__main__":
    audio_process()
