#!/bin/sh

lidar_dataset_path="/home/byungchanchoi/KITTI Dataset/data_odometry_velodyne/dataset/sequences"
image_dataset_path="/home/byungchanchoi/KITTI Dataset/dataset/sequences"
pose_dataset_path="/home/byungchanchoi/KITTI Dataset/data_odometry_poses/dataset/poses"

training_epoch=100
batch_size=16
sequence_length=5
data_display=False

cuda_num='1'

execution_mode="training"
pre_trained_network_path="./1620221072.933818/CRNN_VO_model.pth"

python3 02_RNN_CNN_combination_tutorial.py --input_lidar_file_path "$lidar_dataset_path" \
                                           --input_img_file_path "$image_dataset_path" \
                                           --input_pose_file_path "$pose_dataset_path" \
                                           --cuda_num "$cuda_num" \
                                           --training_epoch "$training_epoch" \
                                           --batch_size "$batch_size" \
                                           --sequence_length "$sequence_length" \
                                           --data_display "$data_display" \
                                           --execution_mode "$execution_mode" \
                                           --pre_trained_network_path "$pre_trained_network_path" \

exit 0