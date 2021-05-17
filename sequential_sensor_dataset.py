import os
import os.path
import numpy as np
import csv

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
from torchvision import transforms

import PIL
from PIL import Image

from lidar_range_img_generator import range_img_generator

class seq_dataset_dict_generator():

    def __init__(self, lidar_dataset_path='', img_dataset_path='', pose_dataset_path='',
                       train_sequence=['00'], valid_sequence=['01'], test_sequence=['02'],
                       sequence_length=1):

        self.lidar_dataset_path = lidar_dataset_path
        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        dataset_sequence_list = np.array([train_sequence, valid_sequence, test_sequence], dtype=np.object)
        
        self.train_dataset_dict_path = './Train_lidar_dataset.csv'
        self.valid_dataset_dict_path = './Valid_lidar_dataset.csv'
        self.test_dataset_dict_path = './Test_lidar_dataset.csv'

        self.train_len = 0
        self.valid_len = 0
        self.test_len = 0

        ######################################
        ### Dataset Dictionary Preparation ###
        ######################################
        dataset_dict_idx = 0
        for dataset_type in dataset_sequence_list:

            self.data_idx = 0

            if dataset_dict_idx == 0:
                self.dataset_dict = open(self.train_dataset_dict_path, 'w', encoding='utf-8', newline='')

            elif dataset_dict_idx == 1:
                self.dataset_dict = open(self.valid_dataset_dict_path, 'w', encoding='utf-8', newline='')

            elif dataset_dict_idx == 2:
                self.dataset_dict = open(self.test_dataset_dict_path, 'w', encoding='utf-8', newline='')

            self.dataset_writer = csv.writer(self.dataset_dict)

            header_idx_list = ['current_index', 'Sequence_index']
            header_seq_data_list = []
            for i in range(sequence_length):
                header_seq_data_list.append('LiDAR [{}] Path'.format(i))
            header_unit_list = ['current_x [m]', 'current_y [m]', 'current_z [m]', 'current_roll [rad]', 'current_pitch [rad]', 'current_yaw [rad]']
            
            header_list = header_idx_list + header_seq_data_list + header_unit_list
            
            self.dataset_writer.writerow(header_list)

            for sequence_idx in np.array(dataset_type):
                
                lidar_base_path = self.lidar_dataset_path + '/' + sequence_idx + '/velodyne'
                lidar_data_name = sorted(os.listdir(lidar_base_path))

                # Pose data accumulation
                lines = []
                pose_file = open(self.pose_dataset_path + '/' + sequence_idx + '.txt', 'r')
                while True:
                    line = pose_file.readline()
                    lines.append(line)
                    if not line: break
                pose_file.close()

                # Iterate over each data in the sequence
                for idx in range(len(lidar_data_name)):
                
                    # If current index is lower than sequence length, skip
                    if idx < sequence_length:
                        pass
                    
                    # If current index is higher than sequence length, accumulate file path into dataset table
                    else:
                        
                        # Latest Pose Retrieval : Pose data re-organization into x, y, z, euler angles
                        current_pose_line = lines[idx]
                        current_pose = current_pose_line.strip().split()

                        current_pose_T = [float(current_pose[3]), float(current_pose[7]), float(current_pose[11])]
                        current_pose_Rmat = np.array([
                                                    [float(current_pose[0]), float(current_pose[1]), float(current_pose[2])],
                                                    [float(current_pose[4]), float(current_pose[5]), float(current_pose[6])],
                                                    [float(current_pose[8]), float(current_pose[9]), float(current_pose[10])]
                                                    ])

                        current_x = current_pose_T[0]
                        current_y = current_pose_T[1]
                        current_z = current_pose_T[2]

                        current_roll = np.arctan2(current_pose_Rmat[2][1], current_pose_Rmat[2][2])
                        current_pitch = np.arctan2(-1 * current_pose_Rmat[2][0], np.sqrt(current_pose_Rmat[2][1]**2 + current_pose_Rmat[2][2]**2))
                        current_yaw = np.arctan2(current_pose_Rmat[1][0], current_pose_Rmat[0][0])

                        # Prev Pose Retrieval : Pose data re-organization into x, y, z, euler angles
                        prev_pose_line = lines[idx-1]
                        prev_pose = prev_pose_line.strip().split()

                        prev_pose_T = [float(prev_pose[3]), float(prev_pose[7]), float(prev_pose[11])]
                        prev_pose_Rmat = np.array([
                                                    [float(prev_pose[0]), float(prev_pose[1]), float(prev_pose[2])],
                                                    [float(prev_pose[4]), float(prev_pose[5]), float(prev_pose[6])],
                                                    [float(prev_pose[8]), float(prev_pose[9]), float(prev_pose[10])]
                                                    ])

                        prev_x = prev_pose_T[0]
                        prev_y = prev_pose_T[1]
                        prev_z = prev_pose_T[2]

                        prev_roll = np.arctan2(prev_pose_Rmat[2][1], prev_pose_Rmat[2][2])
                        prev_pitch = np.arctan2(-1 * prev_pose_Rmat[2][0], np.sqrt(prev_pose_Rmat[2][1]**2 + prev_pose_Rmat[2][2]**2))
                        prev_yaw = np.arctan2(prev_pose_Rmat[1][0], prev_pose_Rmat[0][0])

                        # Pose Change Calculation
                        dx = current_x - prev_x
                        dy = current_y - prev_y
                        dz = current_z - prev_z

                        droll = current_roll - prev_roll
                        dpitch = current_pitch - prev_pitch
                        dyaw = current_yaw - prev_yaw

                        # Accumulate data to form dataset table
                        idx_list = [self.data_idx, sequence_idx]
                        seq_img_path_list = []
                        for idx_offset in reversed(range(sequence_length)):
                            seq_img_path_list.append(lidar_base_path + '/' + lidar_data_name[idx - idx_offset])
                        pose_list = [dx, dy, dz, droll, dpitch, dyaw]

                        data = idx_list + seq_img_path_list + pose_list

                        self.dataset_writer.writerow(data)

                        self.data_idx += 1

            if dataset_dict_idx == 0:
                self.train_len = self.data_idx

            elif dataset_dict_idx == 1:
                self.valid_len = self.data_idx

            elif dataset_dict_idx == 2:
                self.test_len = self.data_idx

            self.dataset_dict.close()

            dataset_dict_idx += 1

class sequential_sensor_dataset(torch.utils.data.Dataset):

    def __init__(self, lidar_dataset_path='', img_dataset_path='', pose_dataset_path='',
                       train_transform=transforms.Compose([]),
                       valid_transform=transforms.Compose([]),
                       test_transform=transforms.Compose([]),
                       train_sequence=['00'], valid_sequence=['01'], test_sequence=['02'],
                       mode='training', normalization=None,
                       sequence_length=1):

        self.lidar_dataset_path = lidar_dataset_path
        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.train_sequence = train_sequence
        self.train_transform = train_transform

        self.valid_sequence = valid_sequence
        self.valid_transform = valid_transform
        
        self.test_sequence = test_sequence
        self.test_transform = test_transform

        self.data_idx = 0

        self.len = 0

        self.mode = mode

        self.sequence_length = sequence_length

        self.lidar_range_img_generator = range_img_generator(h_fov=[-180, 180], h_res=0.2, v_fov=[-24.9, 2], v_res=0.4)

        self.dataset_dict_generator = seq_dataset_dict_generator(lidar_dataset_path=lidar_dataset_path, 
                                                                 img_dataset_path=img_dataset_path, 
                                                                 pose_dataset_path=pose_dataset_path,
                                                                 train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence,
                                                                 sequence_length=self.sequence_length)

        self.train_data_list = []
        train_dataset_dict = open(self.dataset_dict_generator.train_dataset_dict_path, 'r', encoding='utf-8')
        train_reader = csv.reader(train_dataset_dict)
        next(train_reader)     # Skip header row
        for row_data in train_reader:
            self.train_data_list.append(row_data)
        train_dataset_dict.close()

        self.valid_data_list = []
        valid_dataset_dict = open(self.dataset_dict_generator.valid_dataset_dict_path, 'r', encoding='utf-8')
        valid_reader = csv.reader(valid_dataset_dict)
        next(valid_reader)     # Skip heaer row
        for row_data in valid_reader:
            self.valid_data_list.append(row_data)
        valid_dataset_dict.close()

        self.test_data_list = []
        test_dataset_dict = open(self.dataset_dict_generator.test_dataset_dict_path, 'r', encoding='utf-8')
        test_reader = csv.reader(test_dataset_dict)
        next(test_reader)      # Skip header row
        for row_data in test_reader:
            self.test_data_list.append(row_data)
        test_dataset_dict.close()


    ### Invalid data filtering (Exception handling for dataloader) ###
    # Reference : https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/3
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self, index):

        if self.mode == 'training':
            item = self.train_data_list[index]

        elif self.mode == 'validation':
            item = self.valid_data_list[index]

        elif self.mode == 'test':
            item = self.test_data_list[index]

        current_seq = item[1]

        ### Sequential LiDAR Range Image Stacking ###
        for idx in range(self.sequence_length):

            lidar_range_img = self.lidar_range_img_generator.convert_range_img(pcd_path=item[2 + idx], output_type='img_pixel')
            lidar_range_img = np.expand_dims(lidar_range_img, axis=0)   # Add Sequence Length Dimension

            if idx == 0:
                lidar_range_img_stack = lidar_range_img
            else:
                lidar_range_img_stack = np.vstack((lidar_range_img_stack, lidar_range_img))     # Stack the sequence of LiDAR Range Images

        lidar_range_img_stack = torch.from_numpy(lidar_range_img_stack)

        ### Last Time Step 6DOF Pose Data Loading ###
        pose_6DOF = [float(i) for i in item[2 + self.sequence_length:]]
        pose_stack = torch.from_numpy(np.array(pose_6DOF))

        return current_seq, lidar_range_img_stack, pose_stack

    def __len__(self):

        if self.mode == 'training':
            return self.dataset_dict_generator.train_len - 1

        elif self.mode == 'validation':
            return self.dataset_dict_generator.valid_len - 1

        elif self.mode == 'test':
            return self.dataset_dict_generator.test_len - 1

