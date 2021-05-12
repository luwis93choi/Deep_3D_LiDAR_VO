# Reference 01 : KITTI Tutorial (https://github.com/windowsub0406/KITTI_Tutorial)
# Reference 02 : Vehicle Detection from 3D Lidar Using Fully Convolutional Network (http://www.roboticsproceedings.org/rss12/p42.pdf)

import numpy as np
import struct
import open3d as o3d
import cv2 as cv

class range_img_generator():

    def __init__(self, h_fov, h_res, v_fov, v_res, lidar_max_range=120, lidar_min_range=0):

        self.h_fov = h_fov      # LiDAR Horizontal FOV : [Min Angle, Max Angle] / Unit : Degree
        self.h_res = h_res      # LiDAR Horizontal Angular Resolution

        self.v_fov = v_fov      # LiDAR Vertical FOV : [Min Angle, Max Angle] / Unit : Degree
        self.v_res = v_res      # LiDAR Vertical Angular Resolution

        self.min_range = lidar_min_range
        self.max_range = lidar_max_range

    # Directly reading PCD binary file using numpy (Reference : KITTI Tutorial (https://github.com/windowsub0406/KITTI_Tutorial/blob/master/Convert_Velo_2_Pano_detail.ipynb))
    def pcd_from_bin(self, bin_path):

        obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        return obj[:, :3]

    def convert_range_img(self, pcd_path=None, output_type='img_pixel'):

        # In order to convert 3D LiDAR point cloud into 2D range image,
        # each point in point cloud has to be converted / correponded to pixel x, y coordinates in 2D image.

        # In range image, 
        # x coordinate is mapped by horizontal elevation of 3D point
        # y coordinate is mapped by vertical elevation of 3D point

        if pcd_path == None:

            print('[Error] No Data Path')

            return None

        else:

            # pcd = self.convert_bin_to_pcd(bin_path=pcd_path)
            pcd = self.pcd_from_bin(bin_path=pcd_path)

            # Resolution for final range image output
            max_width_steps = int(np.ceil((self.h_fov[1] - self.h_fov[0]) / self.h_res))
            max_height_steps = int(np.ceil((self.v_fov[1] - self.v_fov[0]) / self.v_res))

            h_res_in_radian = self.h_res * (np.pi / 180)    # Horizontal Angular Resolution in Radian
            v_res_in_radian = self.v_res * (np.pi / 180)    # Vertical Angular Resolution in Radian

            pcd_x = pcd[:, 0]
            pcd_y = pcd[:, 1]
            pcd_z = pcd[:, 2]

            pcd_dist_unnormalized = np.sqrt(np.power(pcd_x, 2) + np.power(pcd_y, 2) + np.power(pcd_z, 2))
            pcd_dist_normalized = (pcd_dist_unnormalized - self.min_range) / (self.max_range - self.min_range)

            x_offset = self.h_fov[0] / self.h_res   # Apply offset for negative direction angle / Avoid negative index value
            x_in_range_img = np.arctan2(pcd_y, pcd_x) / h_res_in_radian
            x_in_range_img = np.trunc(x_in_range_img - x_offset).astype(np.int32)
            x_in_range_img = max(x_in_range_img) - x_in_range_img

            y_offset = -1 * self.v_fov[0] / self.v_res  # Apply offset for negative direction angle / Avoid negative index value
            y_in_range_img = np.arcsin(np.divide(pcd_z, pcd_dist_unnormalized)) / v_res_in_radian
            y_in_range_img = np.trunc(y_in_range_img + y_offset).astype(np.int32)
            y_in_range_img = max(y_in_range_img) - y_in_range_img

            # Extra padding for range image representation in order to avoid indexing failure
            # Due to mathematical error offsets that occur during float arithmetic operation, output size of range image varies each time.
            # In order to avoid indexing failure from this, add padding into image.
            img_height_padding = 3
            img_width_padding = 3

            img_height = max(y_in_range_img) + img_height_padding
            img_width = max(x_in_range_img) + img_width_padding

            range_img = np.zeros([img_height, img_width], dtype=np.uint8)

            # Compose range image as the image with pixel value between 0 and 255
            if output_type == 'img_pixel':
                range_img[y_in_range_img, x_in_range_img] = 255 * pcd_dist_normalized

                # Resize the output range image into corrected resolution
                corrected_range_img = cv.resize(range_img, dsize=(320, 200), interpolation=cv.INTER_CUBIC)

                # Add channel dimension in order to support 2D CNN layer's dimension requirements
                corrected_range_img = np.expand_dims(corrected_range_img, axis=0)

            # Compose range image as 2D array with LiDAR scanning range value between minimum and maximum scanning distance
            elif output_type == 'depth':
                range_img[y_in_range_img, x_in_range_img] = pcd_dist_unnormalized
                
                # Resize the output range image into corrected resolution
                corrected_range_img = cv.resize(range_img, dsize=(max_width_steps, max_height_steps), interpolation=cv.INTER_CUBIC)
                
                # Add channel dimension in order to support 2D CNN layer's dimension requirements
                corrected_range_img = np.expand_dims(corrected_range_img, axis=0)

            return corrected_range_img

