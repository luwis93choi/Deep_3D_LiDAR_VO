# Reference : CNN + RNN - Concatenate time distributed CNN with LSTM (https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2)

import os
import sys
import argparse
import cv2 as cv
import numpy as np
import datetime

import torch
from torch import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from sequential_sensor_dataset import sequential_sensor_dataset

from CNN_RNN import CNN_RNN

# Argument parser boolean processing (https://eehoeskrap.tistory.com/521)
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

ap.add_argument('-c', '--cuda_num', type=str, required=True)

ap.add_argument('-e', '--training_epoch', type=int, required=False, default=100)
ap.add_argument('-b', '--batch_size', type=int, required=False, default=16)
ap.add_argument('-s', '--sequence_length', type=int, required=False, default=5)
ap.add_argument('-d', '--data_display', type=str2bool, required=False, default=False)

ap.add_argument('-m', '--execution_mode', type=str, required=True, default='training')
ap.add_argument('-t', '--pre_trained_network_path', type=str, required=True)

args = vars(ap.parse_args())

device = torch.device('cuda:' + args['cuda_num'] if torch.cuda.is_available() else 'cpu')
print(device)

preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.ToTensor(),
])

DATA_DISPLAY_ON = args['data_display']

EPOCH = args['training_epoch']

batch_size = args['batch_size']

sequence_length = args['sequence_length']

mode = args['execution_mode']

dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=['00', '02', '04', '06', '08', '10'], 
                                    valid_sequence=['01', '03', '05', '07', '09'], 
                                    test_sequence=['02'],
                                    sequence_length=sequence_length,
                                    train_transform=preprocess,
                                    valid_transform=preprocess,
                                    test_transform=preprocess,)

start_time = str(datetime.datetime.now())

if mode == 'training':

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=dataset.collate_fn)

    print('Mode : Training')
    print('Training Epoch : ' + str(EPOCH))
    print('Batch Size : ' + str(batch_size))
    print('Sequence Length : ' + str(sequence_length))

    CRNN_VO_model = CNN_RNN(device=device, hidden_size=500, learning_rate=0.001)
    CRNN_VO_model.train()

    # Tensorboard run command : tensorboard --logdir=./runs
    training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/CRNN_LIDAR_VO_training', flush_secs=1)
    validation_writer = SummaryWriter(log_dir='./runs/' + start_time + '/CRNN_LIDAR_VO_validation', flush_secs=1)

    plot_step_training = 0
    plot_step_validation = 0

    for epoch in range(EPOCH):

        print('[EPOCH : {}]'.format(str(epoch)))
        
        ### Training ####################################################################

        dataloader.dataset.mode = 'training'
        CRNN_VO_model.train()

        if epoch == 0:
            if os.path.exists('./' + start_time) == False:
                print('Creating save directory')
                os.mkdir('./' + start_time)

        for batch_idx, (lidar_range_img_stack_tensor, pose_6DOF_tensor) in enumerate(dataloader):

            if (lidar_range_img_stack_tensor != None) and (pose_6DOF_tensor != None):

                lidar_range_img_stack_tensor = lidar_range_img_stack_tensor.to(device).float()
                pose_6DOF_tensor = pose_6DOF_tensor.to(device).float()

                # Data Dimension Standard : Batch Size x Sequence Length x Data Shape
                # Sequential LiDAR Image = Batch Size x Sequence Length x 3 (Channel) x Height x Width
                # Sequential Pose = Batch Size x Sequence Length x 6 (6 DOF)

                # print('---------------------------------')
                # print(lidar_range_img_stack_tensor.size())
                # print(pose_6DOF_tensor.size())

                pose_est_output = CRNN_VO_model(lidar_range_img_stack_tensor)

                translation_rotation_relative_weight = 100

                CRNN_VO_model.optimizer.zero_grad()
                train_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, -1, :3]) \
                            + translation_rotation_relative_weight * CRNN_VO_model.rotation_loss(pose_est_output[:, 3:], pose_6DOF_tensor[:, -1, 3:])
                train_loss.backward()
                CRNN_VO_model.optimizer.step()

                training_writer.add_scalar('Immediate Loss (Translation + Rotation)', train_loss.item(), plot_step_training)
                plot_step_training += 1

                if DATA_DISPLAY_ON is True:

                    ### Sequential Image Stack Display ###
                    disp_current_img_tensor = lidar_range_img_stack_tensor.clone().detach().cpu()

                    lidar_img_sequence_list = []
                    lidar_total_img = []
                    seq_len = dataloader.dataset.sequence_length
                    for batch_index in range(disp_current_img_tensor.size(0)):

                        for seq in range(dataloader.dataset.sequence_length):
                            lidar_range_img = np.array(TF.to_pil_image(disp_current_img_tensor[batch_index][seq]))
                            lidar_range_img = cv.resize(lidar_range_img, dsize=(int(1280/seq_len), int(240/(seq_len * 0.5))), interpolation=cv.INTER_CUBIC)
                            lidar_range_img = cv.applyColorMap(lidar_range_img, cv.COLORMAP_HSV)

                            lidar_img_sequence_list.append(lidar_range_img)

                        lidar_total_img.append(cv.hconcat(lidar_img_sequence_list))
                        lidar_img_sequence_list = []
                    
                    final_lidar_img_output = cv.vconcat(lidar_total_img)

                    cv.imshow('Image Sequence Stack', final_lidar_img_output)
                    cv.waitKey(1)

        ### Validation ###############################################################

        dataloader.dataset.mode = 'validation'
        CRNN_VO_model.eval()

        with torch.no_grad():

            for batch_idx, (lidar_range_img_stack_tensor, pose_6DOF_tensor) in enumerate(dataloader):

                if (lidar_range_img_stack_tensor != None) and (pose_6DOF_tensor != None):

                    lidar_range_img_stack_tensor = lidar_range_img_stack_tensor.to(device).float()
                    pose_6DOF_tensor = pose_6DOF_tensor.to(device).float()

                    # Data Dimension Standard : Batch Size x Sequence Length x Data Shape
                    # Sequential Image = Batch Size x Sequence Length x 3 (Channel) x Height x Width
                    # Sequential Pose = Batch Size x Sequence Length x 6 (6 DOF)

                    # print('---------------------------------')
                    # print(lidar_range_img_stack_tensor.size())
                    # print(pose_6DOF_tensor.size())

                    pose_est_output = CRNN_VO_model(lidar_range_img_stack_tensor)

                    translation_rotation_relative_weight = 100

                    train_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, -1, :3]) \
                                + translation_rotation_relative_weight * CRNN_VO_model.rotation_loss(pose_est_output[:, 3:], pose_6DOF_tensor[:, -1, 3:])

                    validation_writer.add_scalar('Immediate Loss (Translation + Rotation)', train_loss.item(), plot_step_validation)
                    plot_step_validation += 1

        torch.save({
            'epoch' : EPOCH,
            'sequence_length' : sequence_length,
            'CRNN_VO_model' : CRNN_VO_model.state_dict(),
            'optimizer' : CRNN_VO_model.optimizer.state_dict(),
        }, './' + start_time + '/CRNN_VO_model.pth')

elif mode == 'test':

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True, collate_fn=dataset.collate_fn)

    print('Mode : Test')
    print('Batch Size : ' + str(1))
    print('Sequence Length : ' + str(sequence_length))

    CRNN_VO_model = CNN_RNN(device=device, hidden_size=500, learning_rate=0.001)
    CRNN_VO_model.load_state_dict(torch.load(args['pre_trained_network_path'], map_location='cuda:' + args['cuda_num'])['CRNN_VO_model'])
    CRNN_VO_model.eval()

    dataloader.dataset.mode = 'test'
    CRNN_VO_model.eval()

    test_writer = SummaryWriter(log_dir='./runs/' + start_time + '/CRNN_VO_test', flush_secs=1)
    plot_step_test = 0

    with torch.no_grad():

        for batch_idx, (lidar_range_img_stack_tensor, pose_6DOF_tensor) in enumerate(dataloader):

            if (lidar_range_img_stack_tensor != None) and (pose_6DOF_tensor != None):

                lidar_range_img_stack_tensor = lidar_range_img_stack_tensor.to(device).float()
                pose_6DOF_tensor = pose_6DOF_tensor.to(device).float()

                # Data Dimension Standard : Batch Size x Sequence Length x Data Shape
                # Sequential Image = Batch Size x Sequence Length x 3 (Channel) x Height x Width
                # Sequential Pose = Batch Size x Sequence Length x 6 (6 DOF)

                # print('---------------------------------')
                # print(lidar_range_img_stack_tensor.size())
                # print(pose_6DOF_tensor.size())

                pose_est_output = CRNN_VO_model(lidar_range_img_stack_tensor)

                translation_rotation_relative_weight = 100

                train_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, -1, :3]) \
                            + translation_rotation_relative_weight * CRNN_VO_model.rotation_loss(pose_est_output[:, 3:], pose_6DOF_tensor[:, -1, 3:])

                test_writer.add_scalar('Immediate Loss (Translation + Rotation)', train_loss.item(), plot_step_test)
                plot_step_test += 1
