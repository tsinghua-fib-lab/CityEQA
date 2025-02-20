# from vln.env import get_nav_from_actions
# from vln.prompt_builder import get_navigation_lines
import airsim
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import cv2
import sys
import time
# sys.path.append('..')
# from airsim_utils.coord_transformation import quaternion2eularian_angles
# from vln.env import get_nav_from_actions
# from vln.prompt_builder import get_navigation_lines

# xyzw to roll, pitch, yaw
def quaternion2eularian_angles(quat):
    pry = airsim.to_eularian_angles(quat)    # p, r, y
    return np.array([pry[0], pry[1], pry[2]])


AirSimImageType = {
    0: airsim.ImageType.Scene,
    1: airsim.ImageType.DepthPlanar,
    2: airsim.ImageType.DepthPerspective,
    3: airsim.ImageType.DepthVis,
    4: airsim.ImageType.DisparityNormalized,
    5: airsim.ImageType.Segmentation,
    6: airsim.ImageType.SurfaceNormals,
    7: airsim.ImageType.Infrared
}


class AirsimAgent:
    def __init__(self):
        # self.query_func = query_func
        # self.prompt_template = prompt_template
        # self.landmarks = None
        # self.actions = []
        # self.states = []
        # self.cfg = cfg
        self.rotation = R.from_euler("X", -np.pi).as_matrix()
        # self.velocity = 3
        # self.panoid_yaws = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]

        AirSim_client = airsim.MultirotorClient()
        AirSim_client.confirmConnection()
        self.client = AirSim_client

    def world_pos2airsim_pose(self, world_pos):
        airsim_pos = [world_pos[1], world_pos[0], -world_pos[2]]
        return airsim_pos

    def airsim_pos2world_pose(self, airsim_pos):
        world_pos = [airsim_pos[1], airsim_pos[0], -airsim_pos[2]]
        return world_pos

    def setVehiclePose(self, pose: np.ndarray) -> None:
        '''
        pose为[pos, rot]
        rot接受欧拉角或者四元数，
        如果len(pose) == 6,则认为rot为欧拉角,单位为弧度, [pitch, roll, yaw]
        如果len(pose) == 7,则认为rot为四元数, [x, y, z, w]
        '''
        # pose = float(pose)
        pos = self.world_pos2airsim_pose(pose[:3])
        rot = pose[3:]

        if len(rot) == 3:
            rot = np.deg2rad(rot)
            air_rot = airsim.to_quaternion(rot[0], rot[1], rot[2])
        elif len(rot) == 4:
            air_rot = airsim.Quaternionr()
            air_rot.x_val = rot[0]
            air_rot.y_val = rot[1]
            air_rot.z_val = rot[2]
            air_rot.w_val = rot[3]
        else:
            raise ValueError(f"Expected rotation shape is (4,) or (3, ), got ({len(rot)},)")

        air_pos = airsim.Vector3r(pos[0], pos[1], pos[2])
        air_pose = airsim.Pose(air_pos, air_rot)
        self.client.simSetVehiclePose(air_pose, True)
        # self.gt_height = float(air_pos.z_val)
        # print(f"gt z:{self.gt_height}")
        # print(f"set pose: {pos}")

    def get_current_state(self):
        # get world frame pos and orientation
        # orientation is in roll, pitch, yaw format
        state = self.client.simGetGroundTruthKinematics()
        pos = self.airsim_pos2world_pose(state.position.to_numpy_array())
        ori = quaternion2eularian_angles(state.orientation)
        pose = np.concatenate((pos, np.rad2deg(ori)))
        # 将pose的yaw值转化为0-360度
        pose[5] = (pose[5] + 360) % 360
         # pose转化为int型，实现四舍五入
        pose = np.round(pose)
        return pose

    def get_rgb_image(self):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])  # if image_type == 0:
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        # img_out = img_out[:, :, [2, 1, 0]]
        # img_out = response.image_data_uint8

        return img_rgb

    def get_rgbd_image(self):
        # get rgb
        img_rgb = self.get_rgb_image()

        # 获取DepthVis深度可视图
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)])
        img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
        # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
        img_depth_vis = img_depth_planar / 100
        img_depth_vis[img_depth_vis > 1] = 1.
        # 3. 转换为整形
        img_depth = (img_depth_vis * 255).astype(np.uint8)

        return img_rgb, img_depth


