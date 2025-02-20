import copy

import numpy as np
import pickle
import cv2
import os
from sklearn.cluster import DBSCAN
from Utils.arguments import get_args
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from queue import PriorityQueue
from Utils.common_utils import *


def create_extrinsic_matrix(pose):

    position = pose[:3]
    roll_rad = np.deg2rad(pose[3]-90)
    pitch_rad = np.deg2rad(pose[4])
    yaw_rad = np.deg2rad(-pose[5])

    # 计算旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x  # 组合旋转矩阵

    # 创建4x4外参矩阵
    extrinsic_matrix = np.eye(4)

    # 设置旋转部分
    extrinsic_matrix[:3, :3] = R

    # 设置平移部分
    extrinsic_matrix[:3, 3] = position

    return extrinsic_matrix


def create_intrinsic_matrix(args):
    # 计算内参矩阵
    fov = args.camera_fov  # 视场角角度
    width = args.camera_width
    height = args.camera_height

    # 计算焦距
    fx = fy = (width / 2) / np.tan(np.deg2rad(fov) / 2)
    cx = width / 2
    cy = height / 2

    # 内参矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K


def get_next_yaw(current_yaw, move_yaw):
    """
    将当前朝向向移动方向对齐, next_yaw只能等于current_yaw+90或current_yaw-90
    current_yaw: 当前朝向[0, 90, 180, 270]
    move_yaw: 移动方向[0, 90, 180, 270]
    """
    # 计算当前朝向与移动方向的差值
    yaw_diff = move_yaw - current_yaw
    
    # 标准化差值到[-180, 180]范围内
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360
        
    # 根据差值选择旋转方向
    if yaw_diff > 0:
        if yaw_diff <= 90:
            next_yaw = current_yaw + 90
        else:
            next_yaw = current_yaw - 90
    else:
        if yaw_diff >= -90:
            next_yaw = current_yaw - 90
        else:
            next_yaw = current_yaw + 90
            
    # 标准化next_yaw到[0, 360)范围内
    next_yaw = next_yaw % 360
    
    return next_yaw


class CognitiveMap:
    def __init__(self, args):
        """
        Initialize a cognitive map
        """
        # 地图的半径(m)
        self.map_radius = args.map_radius
        # 地图分辨率(m)
        self.map_resolution = args.map_resolution
        # 地图的边长(m)
        self.map_side_length = 2 * self.map_radius
        # 相机高度范围(±)
        self.camera_height_range = args.camera_height_range
        # 相机视野范围(m)
        self.vision_range = args.vision_range
        # 识别物体的阈值
        self.cat_threshold = args.cat_threshold
        # 探索时，检索目标的距离范围(m)
        self.exploration_range = 40
        # 探索时，探测位置之间的距离间隔(m)
        self.exploration_step = 10
        # 导航时，选择可导航网格的距离(m)
        self.navigable_grid_range = 15
        # 导航时，导航位置之间的最大距离间隔(m)
        self.navigation_step_max = 10
        # 导航时，与障碍物保持的期望距离(m)
        self.expected_obstacle_distance = 5
        # 相机内参矩阵
        self.camera_int_matrix = create_intrinsic_matrix(args)

        # 需要初始化的变量
        # 记录物体信息, 每个物体的信息记录如下:"obj_id": {"type": "obj_type","grid": [obj_grid]}
        self.ObjectSet = {}
        # 记录每个网格中的物体信息
        self.GridSet = {}
        # 记录agent的历史轨迹
        self.trajectory = []
        # 导航地图，0：未知，1：已探索，2：障碍物
        self.nav_map = None
        # 绘图句柄
        self.ax = None
        # 相机高度
        self.camera_height = None
        # 地图在世界坐标系下的边界
        self.p_global_origin = None
        self.p_global_max = None

        # 记录物体融合时产生的id编号
        self.obj_merge_id = {}

        # 地图中储存的物体类别（加上drone）
        self.landmark_list = args.landmark_list
        self.landmark_list.append("drone")

    def reset(self, pose):

        self.ObjectSet = {}
        self.GridSet = {}
        self.trajectory = []
        # 导航地图，0：未知，1：已探索，2：障碍物
        self.nav_map = np.zeros((int(self.map_radius * 2 / self.map_resolution), int(self.map_radius * 2 / self.map_resolution)))
        self.ax = None
         
        x = pose[0]
        y = pose[1]
        # 获取地图在世界坐标系下的边界
        self.p_global_origin = (int(x)-self.map_radius, int(y)-self.map_radius)
        self.p_global_max = (int(x)+self.map_radius, int(y)+self.map_radius)
        self.camera_height = pose[2]

        #获取当前pose的网格
        grid_id = self.world_coord_to_grid_id(np.array([[x, y]]))
        # 将当前drone作为物体添加到self.ObjectSet中
        self.ObjectSet["drone"] = {
            "type": "drone",
            "grid": grid_id
        }
        # 在map更新的过程中，landmark的id可能会变化，要记录在这个字典里
        self.obj_merge_id = {}

    def process_new_obs(self, step, depth_image, pose, pred_masks, pred_labels):
        """
        Process new observation

        Args:
            step: 当前步数
            depth_image: 深度图像 H * W
            pose: 相机位置和角度 [x, y, z, roll, pitch, yaw]
            pred_masks: 预测掩码 C * H * W (C为类别数) 当pred_masks[k, h, w] = False时, depth_image[h, w]处的点不属于第k类物体
            pred_labels: 预测标签 长度为C的列表

        """
        # 将uint8格式转化为float格式，还原距离信息
        depth_info = depth_image / 255.0 * 100

        self.trajectory.append(pose)
        # 创建外参矩阵
        camera_ext_matrix = create_extrinsic_matrix(pose)

        # 将深度图转换为世界坐标系下的点云集合
        obj_points_set = self.depth_to_world_coordinates(depth_info, camera_ext_matrix, pred_masks)

        # 显示各个物体类别的点云
        # self.show_obj_points(obj_points_set, pred_labels)

        # 将点云集合转换为栅格地图
        obj_grid_set = self.points_to_grid(obj_points_set)

        # self.show_grid_set(obj_grid_set, pred_labels)

        # 生成物体id，合并物体标签网格信息
        new_obj_set = self.get_obj_set(step, obj_grid_set, pred_labels)

        # 获取网格更新信息
        new_grid_set, new_obj_set = self.get_grid_set(new_obj_set)

        # 将新物体信息和网格更新信息合并到全局物体和网格集合中
        self.update_map(new_obj_set, new_grid_set)

    def depth_to_world_coordinates(self, depth_image, camera_ext_matrix, pred_masks):
        """
        将深度图转换为世界坐标系下的点云坐标,并按pred_masks中顺序输出物体各自的点云集合

        参数:
        depth_image: 深度图像 H * W
        camera_ext_matrix: 相机外参矩阵
        pred_masks: 预测掩码 C * H * W (C为类别数) 当pred_masks[k, h, w] = False时, depth_image[h, w]处的点不属于第k类物体

        返回:
        obj_points_set: 长度为C的列表,每个元素为C个物体的点云集合
        """
        # 获取物体数量和图像尺寸
        obj_num = pred_masks.shape[0]
        height, width = depth_image.shape
        
        # 创建图像坐标网格
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_flat = x.flatten()
        y_flat = y.flatten()
        depth_flat = depth_image.flatten()
        
        # 过滤掉超出视野范围的点
        valid_range = depth_flat <= self.vision_range
        x_flat = x_flat[valid_range]
        y_flat = y_flat[valid_range]
        depth_flat = depth_flat[valid_range]
        
        # 相机内外参矩阵
        K = self.camera_int_matrix
        K_inv = np.linalg.inv(K)
        
        # 计算像素的归一化坐标
        pixel_coords = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)
        
        # 将像素坐标转换为相机坐标系
        camera_coords = K_inv @ pixel_coords
        camera_coords = camera_coords * depth_flat
        
        # 转换为齐次坐标
        camera_coords_homo = np.vstack([camera_coords, np.ones((1, camera_coords.shape[1]))])
        
        # 转换到世界坐标系
        world_coords = camera_ext_matrix @ camera_coords_homo
        world_coords = world_coords[:3, :] / world_coords[3, :]

        # 更新导航地图
        self.update_nav_map(world_coords)

        # 为每个物体创建点云集合
        obj_points_set = []
        for i in range(obj_num):
            mask_flat = pred_masks[i].flatten()[valid_range]  # 应用相同的范围过滤
            obj_points = world_coords[:, mask_flat].T
            # obj_points = self.denoise_points(obj_points)
            obj_points_set.append(obj_points)
            
        return obj_points_set
    
    def denoise_points(self, obj_points):
        """
        使用DBSCAN聚类对点云进行去噪处理

        参数:
        obj_points: 物体点云集合, 维度为[n,3]的数组，包含了n个点云的三维坐标

        返回:
        denoise_points: 去噪后的点云集合    
        """
        # 如果点云数量太少,直接返回
        if len(obj_points) < 10:
            return obj_points
            
        # 使用DBSCAN进行聚类
        # eps为邻域半径,min_samples为成为核心点所需的最小邻域样本数
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(obj_points)
        
        # 获取聚类标签
        labels = clustering.labels_
        
        # 找出最大的聚类(点数最多的聚类)
        unique_labels = np.unique(labels)
        max_cluster_size = 0
        max_cluster_label = None
        
        for label in unique_labels:
            if label == -1:  # -1表示噪声点
                continue
            cluster_size = np.sum(labels == label)
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                max_cluster_label = label
                
        # 保留最大聚类中的点,过滤掉噪声点和其他小聚类
        mask = labels == max_cluster_label
        denoised_points = obj_points[mask]
        
        return denoised_points
        
    def update_nav_map(self, world_points):
        """
        更新导航地图
        取所有点云数据，投影到2D平面，只要在边界范围内的点，都标记为1：已探索
        而后取一定相机高度范围的点云数据，投影到2D平面，取所有在边界范围内的点，标记为2：障碍物

        参数:
        world_points: 点云数据，维度为[3,n]的数组，包含了n个点云的三维坐标
        """
        # 检查输入点云是否为空
        if world_points.size == 0:
            return

        # 先处理所有点云作为已探索区域
        points_2d = world_points[:2, :]
        self._update_grid_map(points_2d, value=1)

        # 处理高度范围内的点云作为障碍物
        mask = ((world_points[2, :] >= self.camera_height - self.camera_height_range) &
                (world_points[2, :] <= self.camera_height + self.camera_height_range))
        obstacle_points = world_points[:2, mask]
        self._update_grid_map(obstacle_points, value=2)

    def _update_grid_map(self, points_2d, value):
        """
        辅助函数：更新网格地图中的指定值
        
        参数:
        points_2d: 2D点云数据，维度为[2,n]
        value: 要更新的网格值
        """
        # 过滤掉超过边界范围的点
        mask = ((points_2d[0, :] >= self.p_global_origin[0]) & (points_2d[0, :] <= self.p_global_max[0]) &
                (points_2d[1, :] >= self.p_global_origin[1]) & (points_2d[1, :] <= self.p_global_max[1]))
        valid_points = points_2d[:, mask]

        if valid_points.size == 0:
            return

        # 将点云坐标转换为网格索引
        grid_coords = np.floor((valid_points - np.array(self.p_global_origin)[:, np.newaxis]) / 
                             self.map_resolution).astype(np.int32)

        # 确保网格索引在合法范围内
        grid_shape = self.nav_map.shape
        valid_mask = ((grid_coords[0, :] >= 0) & (grid_coords[0, :] < grid_shape[0]) &
                     (grid_coords[1, :] >= 0) & (grid_coords[1, :] < grid_shape[1]))
        grid_coords = grid_coords[:, valid_mask]

        # 更新导航地图
        if grid_coords.size > 0:
            self.nav_map[grid_coords[0, :], grid_coords[1, :]] = value

    def points_to_grid(self, obj_points_set):
        """
        将物体的点云集合转换为栅格地图

        参数:
        obj_points_set: 长度为C的列表,每个元素为C个物体的点云集合

        返回:
        grid_set: 长度为C的列表,每个元素为C个物体的栅格编号集合,每个栅格编号为一个整数,表示物体占据的网格编号
        """
        grid_set = []

        for obj_points in obj_points_set:
            # 只取x,y坐标用于2D投影
            points_2d = obj_points[:, :2]

            # 过滤掉超出边界的点
            mask = np.all((points_2d >= self.p_global_origin) & (points_2d <= self.p_global_max), axis=1)
            valid_points = points_2d[mask]

            if len(valid_points) == 0:
                grid_set.append(np.array([]))
                continue

            # 将点云坐标转换为网格索引
            grid_coords = np.floor((valid_points - self.p_global_origin) / self.map_resolution).astype(np.int32)

            # 统计每个网格中的点数
            unique_coords, counts = np.unique(grid_coords, axis=0, return_counts=True)

            # 只保留点数大于阈值的网格
            valid_grids = unique_coords[counts >= self.cat_threshold]

            # 将二维网格坐标转换为一维网格编号
            grid_width = self.map_side_length / self.map_resolution
            grid_nums = valid_grids[:, 1] * grid_width + valid_grids[:, 0]

            grid_set.append(grid_nums)

        return grid_set

    def get_obj_set(self, step, obj_grid_set, pred_labels):
        """
        生成物体id,合并物体标签网格信息

        参数:
        obj_grid_set: 长度为C的列表,每个元素为C个物体的栅格编号集合,每个栅格编号为一个整数,表示物体占据的网格编号
        pred_labels: 长度为C的列表,每个元素为C个物体的标签

        返回:
        new_obj_set: 字典,键为物体id(格式："{step}_{index}")，值为包含物体类型和网格信息的字典
        """
        new_obj_set = {}
        for obj_grid, label in zip(obj_grid_set, pred_labels):
            # 如果obj_grid为空,跳过此次循环
            if len(obj_grid) == 0:
                continue

            # 获取物体类型
            obj_type = label.split('(')[0]  # 提取括号前的物体类型字符串
            
            # 生成唯一id
            obj_id = f"{step}_{len(new_obj_set)}"
            new_obj_set[obj_id] = {
                    "type": obj_type, 
                    "grid": obj_grid
                }

        return new_obj_set

    def get_grid_set(self, obj_set):
        """
        获取网格更新信息
        参数:
        new_obj_set: 字典,键为物体id，值为包含物体类型和网格信息的字典

        返回:
        new_grid_set: 字典，记录每个网格中的新增物体，键为网格编号，值为另一个字典，键为物体类型，值为物体id
        """
        new_obj_set = copy.deepcopy(obj_set)
        new_grid_set = {}
        for obj_id, obj_info in new_obj_set.items():
            for grid_num in obj_info["grid"]:
                # 如果网格编号不在字典中,创建新字典
                if grid_num not in new_grid_set:
                    new_grid_set[grid_num] = {}
                    
                # 如果网格中不存在相同类型的物体,才添加新物体信息
                if obj_info["type"] not in new_grid_set[grid_num]:
                    new_grid_set[grid_num][obj_info["type"]] = obj_id
                else:
                    # 如果网格中已存在同类物体,从物体的网格集合中删除该网格
                    grid_list = new_obj_set[obj_id]["grid"]
                    grid_list = grid_list[grid_list != grid_num]
                    new_obj_set[obj_id]["grid"] = grid_list
                
        return new_grid_set, new_obj_set

    def update_map(self, new_obj_set, new_grid_set):
        """
        将新物体信息和网格更新信息合并到全局物体和网格集合中

        参数:
        new_obj_set: 字典,键为物体id，值为包含物体类型和网格信息的字典
        new_grid_set: 字典，记录每个网格中的新增物体，键为网格编号，值为另一个字典，键为物体类型，值为物体id
        """
        # 首先更新网格集合
        for grid_num, obj_types in new_grid_set.items():
            if grid_num in self.GridSet:
                # 如果网格已存在,合并物体信息
                for obj_type, obj_id in obj_types.items():
                    if obj_type not in self.GridSet[grid_num]:
                        # 如果该类型物体不存在,则添加
                        self.GridSet[grid_num][obj_type] = obj_id
                    else:
                        # 如果该类型物体已存在,从new_obj_set中删除该物体对应的网格
                        if obj_id in new_obj_set:
                            grid_list = new_obj_set[obj_id]["grid"]
                            grid_list = grid_list[grid_list != grid_num]
                            new_obj_set[obj_id]["grid"] = grid_list
            else:
                # 如果网格不存在,直接添加
                self.GridSet[grid_num] = obj_types

        # 更新物体集合
        for obj_id, obj_info in new_obj_set.items():
            if len(obj_info["grid"]) > 0:
                self.ObjectSet[obj_id] = obj_info

        # 对物体进行融合处理
        self.fuse_objs()

    def fuse_objs(self):
        """
        对self.ObjectSet中的物体进行融合处理,同时更新self.GridSet中的物体信息。
        对self.ObjectSet中同种类型的物体进行判断，如果相同类型的物体有网格相邻，则将后创建的物体合并到先创建的物体当中，并且并更新self.GridSet中的物体信息。
        最后检查self.ObjectSet中的物体，如果物体分布在多个区域，这些区域之间没有相邻的网格，则仅保留相邻网格数量最多的区域。
        """
        # 清空之前的物体融合记录
        self.obj_merge_id = {}
        # 按物体类型分组
        obj_by_type = {}
        for obj_id, obj_info in self.ObjectSet.items():
            if obj_id == "road" or obj_id == "drone":
                continue
            obj_type = obj_info["type"]
            if obj_type not in obj_by_type:
                obj_by_type[obj_type] = []
            obj_by_type[obj_type].append(obj_id)
        
        # 对每种类型的物体进行融合
        for obj_type, obj_ids in obj_by_type.items():
            if len(obj_ids) <= 1:
                continue
                
            # 检查每对物体是否相邻
            i = 0
            while i < len(obj_ids):
                obj1_id = obj_ids[i]
                if obj1_id not in self.ObjectSet:
                    i += 1
                    continue
                    
                obj1_grids = self.ObjectSet[obj1_id]["grid"]
                j = i + 1
                
                while j < len(obj_ids):
                    obj2_id = obj_ids[j]
                    if obj1_id == obj2_id:
                        j += 1
                        continue
                    if obj2_id not in self.ObjectSet:
                        j += 1
                        continue
                        
                    obj2_grids = self.ObjectSet[obj2_id]["grid"]
                    
                    # 检查两个物体的网格是否相邻
                    grid_coords1 = self.grid_id_to_local_coord(obj1_grids)
                    grid_coords2 = self.grid_id_to_local_coord(obj2_grids)
                    
                    # 使用广播机制计算所有网格点对之间的距离
                    # grid_coords1: (N,2), grid_coords2: (M,2) -> dists: (N,M)
                    dists = np.sqrt(np.sum((grid_coords1[:,np.newaxis,:] - grid_coords2[np.newaxis,:,:]) ** 2, axis=2))
                    # 获取最小距离
                    min_dist = np.min(dists)
                            
                    if min_dist <= 1:  # 物体相邻
                        # 将后创建的物体合并到先创建的物体，物体创建的先后可通过obj_ids的顺序来判断
                        main_obj = obj1_id # 保留的物体id
                        merged_obj = obj2_id # 被合并的物体id
                        self.obj_merge_id[merged_obj] = main_obj
                            
                        # 更新ObjectSet
                        merged_grids = self.ObjectSet[merged_obj]["grid"]
                        self.ObjectSet[main_obj]["grid"] = np.unique(np.concatenate([
                            self.ObjectSet[main_obj]["grid"],
                            merged_grids
                        ]))
                        
                        # 更新GridSet
                        for grid_num in merged_grids:
                            if grid_num in self.GridSet and obj_type in self.GridSet[grid_num]:
                                self.GridSet[grid_num][obj_type] = main_obj
                                
                        # 删除被合并的物体
                        del self.ObjectSet[merged_obj]
                        obj_ids.remove(merged_obj)
                        # 在obj_ids的最后加上main_obj，因为main_obj可能还能和其他物体合并
                        obj_ids.append(main_obj)
                        continue
                        
                    j += 1
                i += 1
                
            # 检查每个物体的网格连通性,只保留最大的连通区域
            # for obj_id in obj_ids:
            #     if obj_id not in self.ObjectSet:
            #         continue
            #
            #     grid_coords = self.grid_id_to_local_coord(self.ObjectSet[obj_id]["grid"])
            #
            #     # 使用DBSCAN聚类找出连通区域
            #     from sklearn.cluster import DBSCAN
            #     clustering = DBSCAN(eps=1.5, min_samples=1).fit(grid_coords)
            #     labels = clustering.labels_
            #
            #     # 找出最大的连通区域
            #     unique_labels, counts = np.unique(labels, return_counts=True)
            #     max_cluster = unique_labels[np.argmax(counts)]
            #
            #     # 只保留最大连通区域的网格
            #     mask = (labels == max_cluster)
            #     kept_coords = grid_coords[mask]
            #     kept_grids = self.local_coord_to_grid_id(kept_coords).astype(np.int32)
            #
            #     # 更新ObjectSet
            #     self.ObjectSet[obj_id]["grid"] = kept_grids
            #
            #     # 更新GridSet,删除不在最大连通区域的网格中的物体引用
            #     all_grids = self.ObjectSet[obj_id]["grid"]
            #     removed_grids = all_grids[~np.isin(all_grids, kept_grids)]
            #     for grid_num in removed_grids:
            #         if grid_num in self.GridSet and obj_type in self.GridSet[grid_num]:
            #             del self.GridSet[grid_num][obj_type]
            #             if not self.GridSet[grid_num]:  # 如果网格中没有其他物体了
            #                 del self.GridSet[grid_num]

    ## 坐标转化相关函数
    def grid_id_to_local_coord(self, grid_set):
        """
        将网格编号转换为网格坐标(在cogmap中的坐标，而不是世界坐标)
        
        参数:
        grid_set: numpy.ndarray, 形状为(N,)的数组,每个元素为网格编号
        
        返回:
        coords: numpy.ndarray, 形状为(N,2)的数组,每行为一个网格的[x,y]坐标
        """
        grid_width = self.map_side_length / self.map_resolution
        x = grid_set % grid_width
        y = grid_set // grid_width
        coords = np.column_stack((x, y))
        return coords

    def world_coord_to_grid_id(self, grid_coords):
        """
        将世界坐标转换为网格编号
        参数:
        grid_coords: numpy.ndarray, 形状为(N,2)的数组,每行为一个网格的[x,y]坐标
        返回:
        grid_id: numpy.ndarray, 形状为(N,)的数组,每个元素为网格编号
        """
        grid_width = self.map_side_length / self.map_resolution
        # 将世界坐标转换为网格编号
        grid_x = np.floor((grid_coords[:, 0] - self.p_global_origin[0]) / self.map_resolution).astype(np.int32)
        grid_y = np.floor((grid_coords[:, 1] - self.p_global_origin[1]) / self.map_resolution).astype(np.int32)
        
        # 计算一维网格编号
        grid_id = grid_y * grid_width + grid_x
        
        return grid_id

    def local_coord_to_grid_id(self, grid_coords):
        """
        将局部坐标转换为网格编号
        参数:
        grid_coords: numpy.ndarray, 形状为(N,2)的数组,每行为一个网格的[x,y]坐标
        返回:
        grid_id: numpy.ndarray, 形状为(N,)的数组,每个元素为网格编号
        """
        grid_width = self.map_side_length / self.map_resolution
        grid_id = grid_coords[:, 1] * grid_width + grid_coords[:, 0]
        return grid_id

    ## 显示各种地图
    def show_grid_set(self, grid_set, pred_labels):
        """
        显示各个物体类别的栅格地图

        参数:
        grid_set: 长度为C的列表,每个元素为N个网格编号
        pred_labels: 长度为C的列表,每个元素为字符串,记录物体的名称
        """
        # 创建图形
        plt.figure(figsize=(10, 10))

        # 为每个物体分配不同的颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(grid_set)))

        # 绘制每个物体的栅格
        for grids, label, color in zip(grid_set, pred_labels, colors):
            if len(grids) > 0:
                # 将网格编号转换为网格坐标
                grid_coords = self.grid_id_to_local_coord(grids)
                
                # 将栅格坐标转换为实际坐标(米)
                real_coords = grid_coords * self.map_resolution + self.p_global_origin

                # 绘制每个栅格
                for x, y in real_coords:
                    # 创建一个resolution大小的方形网格
                    rect = plt.Rectangle((x, y),
                                      self.map_resolution, self.map_resolution,
                                      facecolor=color[:3],
                                      alpha=0.6,
                                      label=label if (x == real_coords[0][0] and y == real_coords[0][1]) else "")
                    plt.gca().add_patch(rect)

                # 设置显示范围
                plt.gca().set_xlim(self.p_global_origin[0], self.p_global_max[0])
                plt.gca().set_ylim(self.p_global_origin[1], self.p_global_max[1])

        # 添加网格线
        x_grid = np.arange(self.p_global_origin[0], self.p_global_max[0] + self.map_resolution, self.map_resolution)
        y_grid = np.arange(self.p_global_origin[1], self.p_global_max[1] + self.map_resolution, self.map_resolution)
        plt.grid(True, which='major', linestyle='-', alpha=0.2)
        plt.gca().set_xticks(x_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度
        plt.gca().set_yticks(y_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.axis('equal')  # 保持横纵比例相等
        plt.show()

    def show_obj_points(self, obj_points_set, pred_labels):

        # 创建图形
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 为每个物体分配不同的颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(obj_points_set)))

        # 绘制每个物体的点云
        for points, label, color in zip(obj_points_set, pred_labels, colors):
            if len(points) > 0:
                # 随机采样点以减少显示数量（可选）
                if len(points) > 1000:
                    idx = np.random.choice(len(points), 1000, replace=False)
                    points = points[idx]

                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=[color[:3]], label=label, alpha=0.6, s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    
    def show_nav_map(self):
        """
        显示导航地图
        """
        if self.ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            self.ax = ax
        else:
            self.ax.clear()
        
        # 绘制导航地图
        self.ax.imshow(self.nav_map.T, cmap='gray_r', 
                      extent=[self.p_global_origin[0], self.p_global_max[0], 
                              self.p_global_origin[1], self.p_global_max[1]], 
                      origin='lower')  # 使用origin='lower'来确保y轴方向正确

        # 添加网格线
        x_grid = np.arange(self.p_global_origin[0], self.p_global_max[0] + self.map_resolution, self.map_resolution)
        y_grid = np.arange(self.p_global_origin[1], self.p_global_max[1] + self.map_resolution, self.map_resolution)
        self.ax.grid(True, which='major', linestyle='-', alpha=0.2)
        self.ax.set_xticks(x_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度
        self.ax.set_yticks(y_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Navigation Map')
        self.ax.axis('equal')  # 保持横纵比例相等
        plt.draw()  # 更新图像
        plt.pause(0.1)  # 暂停以更新图像

    def show_cogmap(self):
        """
        显示认知地图，根据self.ObjectSet中的物体分布信息，绘制栅格地图
        self.ObjectSet: 字典,键为物体id，值为包含物体类型和网格信息的字典
        每个物体的名称由物体id和物体类型组成，格式为"{id}_{type}"
        """
        if self.ax is None:
            # 创建图形和轴对象
            fig, ax = plt.subplots(figsize=(10, 10))
            self.ax = ax  # 保存轴对象以便后续更新
        else:
            self.ax.clear()  # 清除当前轴内容以便更新

        # 获取物体数量并为每个物体分配不同的颜色
        obj_num = len(self.ObjectSet)
        colors = plt.cm.rainbow(np.linspace(0, 1, obj_num))

        # 遍历每个物体
        for (obj_id, obj_info), color in zip(self.ObjectSet.items(), colors):
            grids = obj_info['grid']
            obj_type = obj_info['type']

            if len(grids) > 0:
                # 将网格编号转换为网格坐标
                grid_coords = self.grid_id_to_local_coord(grids)

                # 将栅格坐标转换为实际坐标(米)
                real_coords = grid_coords * self.map_resolution + self.p_global_origin

                # 生成物体标签
                label = f"{obj_type}_{obj_id}"

                # 绘制每个栅格
                for x, y in real_coords:
                    # 创建一个resolution大小的方形网格
                    rect = plt.Rectangle((x, y),
                                         self.map_resolution, self.map_resolution,
                                         facecolor=color[:3],
                                         alpha=0.6,
                                         label=label if (x == real_coords[0][0] and y == real_coords[0][1]) else "")
                    self.ax.add_patch(rect)

        # 绘制agent位置和朝向
        agent_pos = self.trajectory[-1]
        agent_x, agent_y, _, _, _, agent_yaw = agent_pos
        # 计算三角形的三个顶点
        # 三角形大小
        triangle_size = 2
        # 将角度转换为弧度，0度朝向y轴正方向，90度朝向x轴正方向
        angle_rad = np.deg2rad(agent_yaw)
        # 三角形顶点 - 使用更细长的三角形
        point1 = [agent_x + triangle_size * 1 * np.sin(angle_rad),
                 agent_y + triangle_size * 1 * np.cos(angle_rad)]
        point2 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad + np.deg2rad(140)),
                 agent_y + triangle_size * 0.5 * np.cos(angle_rad + np.deg2rad(140))]
        point3 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad - np.deg2rad(140)),
                 agent_y + triangle_size * 0.5 * np.cos(angle_rad - np.deg2rad(140))]
        triangle = plt.Polygon([point1, point2, point3], color='r', label='Agent')
        self.ax.add_patch(triangle)

        # 绘制agent的历史轨迹
        trajectory = np.array(self.trajectory)
        if trajectory.size > 0:
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=0.8, label='Trajectory')

        # 设置显示范围
        self.ax.set_xlim(self.p_global_origin[0], self.p_global_max[0])
        self.ax.set_ylim(self.p_global_origin[1], self.p_global_max[1])

        # 添加网格线
        x_grid = np.arange(self.p_global_origin[0], self.p_global_max[0] + self.map_resolution, self.map_resolution)
        y_grid = np.arange(self.p_global_origin[1], self.p_global_max[1] + self.map_resolution, self.map_resolution)
        self.ax.grid(True, which='major', linestyle='-', alpha=0.2)
        self.ax.set_xticks(x_grid[::int(20 / self.map_resolution)])  # 每20m显示一个刻度
        self.ax.set_yticks(y_grid[::int(20 / self.map_resolution)])  # 每20m显示一个刻度

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.legend()
        self.ax.set_aspect('equal')  # 保持横纵比例相等
        plt.draw()  # 更新图像
        plt.pause(0.1)  # 暂停以更新图像

    def show_map(self):
        """
        显示地图所有要素，首先显示导航地图，然后将认知地图中的物体绘制在导航地图上
        """
        """
        显示导航地图
        """
        if self.ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            self.ax = ax
        else:
            self.ax.clear()
        
        # 绘制导航地图
        self.ax.imshow(self.nav_map.T, cmap='gray_r',
                      extent=[self.p_global_origin[0], self.p_global_max[0], 
                              self.p_global_origin[1], self.p_global_max[1]], 
                      origin='lower')  # 使用origin='lower'来确保y轴方向正确

        # 绘制认知地图
        # 获取物体数量并为每个物体分配不同的颜色
        obj_num = len(self.ObjectSet)
        colors = plt.cm.rainbow(np.linspace(0, 1, obj_num))

        # 遍历每个物体
        for (obj_id, obj_info), color in zip(self.ObjectSet.items(), colors):
            grids = obj_info['grid']
            obj_type = obj_info['type']

            if len(grids) > 0:
                # 将网格编号转换为网格坐标
                grid_coords = self.grid_id_to_local_coord(grids)

                # 将栅格坐标转换为实际坐标(米)
                real_coords = grid_coords * self.map_resolution + self.p_global_origin

                # 生成物体标签
                label = f"{obj_type}_{obj_id}"

                # 绘制每个栅格
                for x, y in real_coords:
                    # 创建一个resolution大小的方形网格
                    rect = plt.Rectangle((x, y),
                                         self.map_resolution, self.map_resolution,
                                         facecolor=color[:3],
                                         alpha=0.6,
                                         label=label if (x == real_coords[0][0] and y == real_coords[0][1]) else "")
                    self.ax.add_patch(rect)

        # 绘制agent位置和朝向
        agent_pos = self.trajectory[-1]
        agent_x, agent_y, _, _, _, agent_yaw = agent_pos
        # 计算三角形的三个顶点
        # 三角形大小
        triangle_size = 2
        # 将角度转换为弧度，0度朝向y轴正方向，90度朝向x轴正方向
        angle_rad = np.deg2rad(agent_yaw)
        # 三角形顶点 - 使用更细长的三角形
        point1 = [agent_x + triangle_size * 1 * np.sin(angle_rad),
                 agent_y + triangle_size * 1 * np.cos(angle_rad)]
        point2 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad + np.deg2rad(140)),
                 agent_y + triangle_size * 0.5 * np.cos(angle_rad + np.deg2rad(140))]
        point3 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad - np.deg2rad(140)),
                 agent_y + triangle_size * 0.5 * np.cos(angle_rad - np.deg2rad(140))]
        triangle = plt.Polygon([point1, point2, point3], color='r', label='Agent')
        self.ax.add_patch(triangle)

        # 绘制agent的历史轨迹
        trajectory = np.array(self.trajectory)
        if trajectory.size > 0:
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=0.8, label='Trajectory')

        # 添加网格线
        x_grid = np.arange(self.p_global_origin[0], self.p_global_max[0] + self.map_resolution, self.map_resolution)
        y_grid = np.arange(self.p_global_origin[1], self.p_global_max[1] + self.map_resolution, self.map_resolution)
        self.ax.grid(True, which='major', linestyle='-', alpha=0.2)
        self.ax.set_xticks(x_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度
        self.ax.set_yticks(y_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Map')
        self.ax.legend()
        self.ax.axis('equal')  # 保持横纵比例相等
        plt.draw()  # 更新图像
        plt.pause(0.1)  # 暂停以更新图像

    ## 保存地图
    def save_map(self, save_path):
        """
        保存地图所有要素
        """
        # 创建新的图像
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制导航地图
        ax.imshow(self.nav_map.T, cmap='gray_r',
                      extent=[self.p_global_origin[0], self.p_global_max[0], 
                              self.p_global_origin[1], self.p_global_max[1]], 
                      origin='lower')  # 使用origin='lower'来确保y轴方向正确

        # 绘制认知地图
        # 获取物体数量并为每个物体分配不同的颜色
        obj_num = len(self.ObjectSet)
        colors = plt.cm.rainbow(np.linspace(0, 1, obj_num))

        # 遍历每个物体
        for (obj_id, obj_info), color in zip(self.ObjectSet.items(), colors):
            grids = obj_info['grid']
            obj_type = obj_info['type']

            if len(grids) > 0:
                # 将网格编号转换为网格坐标
                grid_coords = self.grid_id_to_local_coord(grids)

                # 将栅格坐标转换为实际坐标(米)
                real_coords = grid_coords * self.map_resolution + self.p_global_origin

                # 生成物体标签
                label = f"{obj_type}_{obj_id}"

                # 绘制每个栅格
                for x, y in real_coords:
                    # 创建一个resolution大小的方形网格
                    rect = plt.Rectangle((x, y),
                                         self.map_resolution, self.map_resolution,
                                         facecolor=color[:3],
                                         alpha=0.6,
                                         label=label if (x == real_coords[0][0] and y == real_coords[0][1]) else "")
                    ax.add_patch(rect)

        # 绘制agent位置和朝向
        agent_pos = self.trajectory[-1]
        agent_x, agent_y, _, _, _, agent_yaw = agent_pos
        # 计算三角形的三个顶点
        # 三角形大小
        triangle_size = 2
        # 将角度转换为弧度，0度朝向y轴正方向，90度朝向x轴正方向
        angle_rad = np.deg2rad(agent_yaw)
        # 三角形顶点 - 使用更细长的三角形
        point1 = [agent_x + triangle_size * 1 * np.sin(angle_rad),
                 agent_y + triangle_size * 1 * np.cos(angle_rad)]
        point2 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad + np.deg2rad(140)),
                 agent_y + triangle_size * 0.5 * np.cos(angle_rad + np.deg2rad(140))]
        point3 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad - np.deg2rad(140)),
                 agent_y + triangle_size * 0.5 * np.cos(angle_rad - np.deg2rad(140))]
        triangle = plt.Polygon([point1, point2, point3], color='r', label='Agent')
        ax.add_patch(triangle)

        # 绘制agent的历史轨迹
        trajectory = np.array(self.trajectory)
        if trajectory.size > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=0.8, label='Trajectory')

        # 添加网格线
        x_grid = np.arange(self.p_global_origin[0], self.p_global_max[0] + self.map_resolution, self.map_resolution)
        y_grid = np.arange(self.p_global_origin[1], self.p_global_max[1] + self.map_resolution, self.map_resolution)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.set_xticks(x_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度
        ax.set_yticks(y_grid[::int(20/self.map_resolution)])  # 每20m显示一个刻度

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Map')
        ax.legend()
        ax.axis('equal')  # 保持横纵比例相等
        
        # 关闭交互模式
        plt.ioff()
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # 关闭图像
        plt.close(fig)

    def save_map_landmark(self, save_path, landmark_id):
        """
        保存地图所有要素
        """
        # 创建新的图像
        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制导航地图
        ax.imshow(self.nav_map.T, cmap='gray_r',
                  extent=[self.p_global_origin[0], self.p_global_max[0],
                          self.p_global_origin[1], self.p_global_max[1]],
                  origin='lower')  # 使用origin='lower'来确保y轴方向正确

        # 绘制认知地图
        # 指定三种颜色
        landmark_color = '#ff7f0e'  # 和谐的橙色
        other_object_color = '#9467bd'  # 和谐的紫色
        # start_pose_color = '#8c564b'  # 和谐的棕色

        # 判断landmark_id是否为None
        if landmark_id is not None and landmark_id in self.ObjectSet:
            obj_info = self.ObjectSet[landmark_id]
            color = landmark_color  # 假设第二个颜色用于landmark
            grids = obj_info['grid']
            # obj_type = obj_info['type']

            if len(grids) > 0:
                grid_coords = self.grid_id_to_local_coord(grids)
                real_coords = grid_coords * self.map_resolution + self.p_global_origin
                label = f"Landmark"

                for x, y in real_coords:
                    rect = plt.Rectangle((x, y),
                                         self.map_resolution, self.map_resolution,
                                         facecolor=color,
                                         alpha=0.6,
                                         label=label if (x == real_coords[0][0] and y == real_coords[0][1]) else "")
                    ax.add_patch(rect)

        # 绘制其余id的物体
        for obj_id, obj_info in self.ObjectSet.items():
            if obj_id == "drone" or obj_id == landmark_id:
                continue

            color = other_object_color
            grids = obj_info['grid']
            obj_type = obj_info['type']

            if len(grids) > 0:
                grid_coords = self.grid_id_to_local_coord(grids)
                real_coords = grid_coords * self.map_resolution + self.p_global_origin
                label = f"Other_object"

                for x, y in real_coords:
                    rect = plt.Rectangle((x, y),
                                         self.map_resolution, self.map_resolution,
                                         facecolor=color,
                                         alpha=0.6,
                                         label=label if (x == real_coords[0][0] and y == real_coords[0][1]) else "")
                    ax.add_patch(rect)

        # 绘制agent位置和朝向
        agent_pos = self.trajectory[-1]
        agent_x, agent_y, _, _, _, agent_yaw = agent_pos
        # 计算三角形的三个顶点
        # 三角形大小
        triangle_size = 2
        # 将角度转换为弧度，0度朝向y轴正方向，90度朝向x轴正方向
        angle_rad = np.deg2rad(agent_yaw)
        # 三角形顶点 - 使用更细长的三角形
        point1 = [agent_x + triangle_size * 1 * np.sin(angle_rad),
                  agent_y + triangle_size * 1 * np.cos(angle_rad)]
        point2 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad + np.deg2rad(140)),
                  agent_y + triangle_size * 0.5 * np.cos(angle_rad + np.deg2rad(140))]
        point3 = [agent_x + triangle_size * 0.5 * np.sin(angle_rad - np.deg2rad(140)),
                  agent_y + triangle_size * 0.5 * np.cos(angle_rad - np.deg2rad(140))]
        triangle = plt.Polygon([point1, point2, point3], color='r', label='Agent')
        ax.add_patch(triangle)

        # 绘制agent的历史轨迹
        trajectory = np.array(self.trajectory)
        if trajectory.size > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=0.8, label='Trajectory')

        # 添加网格线
        x_grid = np.arange(self.p_global_origin[0], self.p_global_max[0] + self.map_resolution, self.map_resolution)
        y_grid = np.arange(self.p_global_origin[1], self.p_global_max[1] + self.map_resolution, self.map_resolution)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.set_xticks(x_grid[::int(20 / self.map_resolution)])  # 每20m显示一个刻度
        ax.set_yticks(y_grid[::int(20 / self.map_resolution)])  # 每20m显示一个刻度

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # ax.set_title('Map')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
            elif label == "Other_object":
                continue

        # 增加Exploded area和Obstacles图例
        exploded_area_patch = plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=1.0, label="Exploded area")
        obstacles_patch = plt.Rectangle((0, 0), 1, 1, facecolor="black", alpha=1.0, label="Obstacles")
        unique_handles.append(exploded_area_patch)
        unique_labels.append("Explored area")
        unique_handles.append(obstacles_patch)
        unique_labels.append("Obstacles")

        ax.legend(unique_handles, unique_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(unique_labels))
        ax.axis('equal')  # 保持横纵比例相等

        # 关闭交互模式
        plt.ioff()
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # 关闭图像
        plt.close(fig)

    ## 以下为探索相关函数
    def get_obj_exp_area_in_direction(self, obj_id, direction):
        """
        获取物体obj_id特定方向的探索区域的顶点坐标

        Args:
            obj_id: 物体id
            direction: 方向，0：北，1：东，2：南，3：西

        Returns:
            exp_area: 探索区域的顶点坐标[x0, y0, x1, y1]
        """
        # 获取物体obj_id的栅格坐标并转换为numpy数组
        obj_grid = self.ObjectSet[obj_id]["grid"]
        coords = np.array(self.grid_id_to_local_coord(obj_grid))

        if direction == 0:  # 北方
            x_coords = np.unique(coords[:, 0])
            y_values = [np.max(coords[coords[:, 0] == x, 1]) for x in x_coords]  # 每列最大y值
            y_min = np.median(y_values)  # 下边界
            x_min, x_max = np.min(x_coords), np.max(x_coords)  # x方向范围
            y_max = y_min + self.exploration_range  # 上边界
            exp_area = [x_min, y_min, x_max, y_max]
            
        elif direction == 1:  # 东方
            y_coords = np.unique(coords[:, 1])
            x_values = [np.min(coords[coords[:, 1] == y, 0]) for y in y_coords]  # 每行最大x值
            x_min = np.median(x_values)  # 左边界
            y_min, y_max = np.min(y_coords), np.max(y_coords)  # y方向范围
            x_max = x_min + self.exploration_range  # 右边界
            exp_area = [x_min, y_min, x_max, y_max]
            
        elif direction == 2:  # 南方
            x_coords = np.unique(coords[:, 0])
            y_values = [np.min(coords[coords[:, 0] == x, 1]) for x in x_coords]  # 每列最小y值
            y_max = np.median(y_values)  # 上边界
            x_min, x_max = np.min(x_coords), np.max(x_coords)  # x方向范围
            y_min = y_max - self.exploration_range  # 下边界
            exp_area = [x_min, y_min, x_max, y_max]

        elif direction == 3:  # 西方
            y_coords = np.unique(coords[:, 1])
            x_values = [np.max(coords[coords[:, 1] == y, 0]) for y in y_coords]  # 每行最小x值
            x_max = np.median(x_values)  # 右边界
            y_min, y_max = np.min(y_coords), np.max(y_coords)  # y方向范围
            x_min = x_max - self.exploration_range  # 左边界
            exp_area = [x_min, y_min, x_max, y_max]

        # 控制探索范围不超过地图边界
        exp_area = [max(exp_area[0], 0),
                    max(exp_area[1], 0),
                    min(exp_area[2], self.map_side_length),
                    min(exp_area[3], self.map_side_length)]
        return exp_area

    def check_target_appeared(self, target_class, landmark_id, direction):
        """
        判断landmark_id的direction方向是否出现了target_class类别的物体，
        范围为self.exploration_range

        Args:
            target_class: 目标物体类别
            landmark_id: 参照物体id
            direction: 探索方向，0：北，1：东，2：南，3：西

        Returns:
            target_obj_id: 重叠面积最大的目标物体id，如果没有重叠则返回None
        """
        # 获取landmark_id的探索区域
        landmark_exp_area = self.get_obj_exp_area_in_direction(landmark_id, direction)

        # 获取所有符合条件的目标物体id
        target_obj_set = {
            obj_id for obj_id, obj_info in self.ObjectSet.items() 
            if obj_info['type'] == target_class and obj_id != landmark_id
        }

        if not target_obj_set:
            return None

        # 计算每个目标物体与探索区域的重叠程度
        overlaps = {}
        for obj_id in target_obj_set:
            coords = np.array(self.grid_id_to_local_coord(self.ObjectSet[obj_id]["grid"]))
            # 使用向量化操作计算重叠
            in_range = (coords >= landmark_exp_area[:2]) & (coords <= landmark_exp_area[2:])
            overlap = np.sum(np.all(in_range, axis=1))
            if overlap > 0:
                overlaps[obj_id] = overlap

        # 返回重叠最大的物体id，如果没有重叠则返回None
        return max(overlaps.items(), key=lambda x: x[1])[0] if overlaps else None

    def get_next_exploration_pose(self, explored_poses, landmark_id, direction, explored_back_flag):
        """
        获取下一个探索点

        Args:
            explored_poses: 已探索的点坐标(世界坐标)，列表，每个元素为[x, y, z, roll, pitch, yaw]
            landmark_id: 参照物体id
            direction: 探索方向，0：北，1：西，2：南，3：东
            explored_back_flag: 折返探索标志，0：向预定的方向探索，1：向相反方向探索

        Returns:
            next_pose: 下一个探索点
        """
        # 获取参照物体的待探索区域
        landmark_exp_area = self.get_obj_exp_area_in_direction(landmark_id, direction)

        # 获取已探索区域中心网格编号
        explored_grids = self.world_coord_to_grid_id(np.array(explored_poses)[:, :2])

        current_grid = explored_grids[-1]
        current_grid_coord = self.grid_id_to_local_coord(current_grid)[0]


        if direction == 0:  # 北方
            # 如果explored_grids 只包含了一个网格，则先向西探索（设定的规则，后续可优化）
            if len(explored_grids) == 1:
                next_pose = explored_poses[-1].copy()
                next_pose[0] = next_pose[0] - self.exploration_step
                return next_pose, explored_back_flag
            else:
                if explored_back_flag == 0:
                    # 当前是向西探索
                    # 如果当前pose位于待探索区域西侧，说明需要向东探索
                    if current_grid_coord[0] < landmark_exp_area[0]:
                        # 取初始探索点东侧的网格坐标
                        next_pose = explored_poses[0].copy()
                        next_pose[0] = next_pose[0] + self.exploration_step
                        explored_back_flag = 1
                        return next_pose, explored_back_flag
                    else:
                        # 如果没到西侧，则继续向西探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[0] = next_pose[0] - self.exploration_step
                        return next_pose, explored_back_flag
                else:
                    # 当前是向东探索
                    # 如果当前pose位于待探索区域东侧，说明所有区域已经探索完毕，返回探索点None
                    if current_grid_coord[0] > landmark_exp_area[2]:
                        return None, explored_back_flag
                    else:
                        # 如果没到东侧，则继续向东探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[0] = next_pose[0] + self.exploration_step
                        return next_pose, explored_back_flag
        elif direction == 1:  # 东方
            # 如果explored_grids 只包含了一个网格，则先向北探索（设定的规则，后续可优化）
            if len(explored_grids) == 1:
                next_pose = explored_poses[-1].copy()
                next_pose[1] = next_pose[1] - self.exploration_step
                return next_pose, explored_back_flag
            else:
                if explored_back_flag == 0:
                    # 当前是向北探索
                    # 如果当前pose位于待探索区域北侧，说明需要向南探索
                    if current_grid_coord[1] < landmark_exp_area[1]:
                        # 取初始探索点南侧的网格坐标
                        next_pose = explored_poses[0].copy()
                        next_pose[1] = next_pose[1] + self.exploration_step
                        explored_back_flag = 1
                        return next_pose, explored_back_flag
                    else:
                        # 如果没到北侧，则继续向北探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[1] = next_pose[1] - self.exploration_step
                        return next_pose, explored_back_flag
                else:
                    # 当前是向南探索
                    # 如果当前pose位于待探索区域南侧，说明所有区域已经探索完毕，返回探索点None
                    if current_grid_coord[1] > landmark_exp_area[3]:
                        return None, explored_back_flag
                    else:
                        # 如果没到南侧，则继续向南探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[1] = next_pose[1] + self.exploration_step
                        return next_pose, explored_back_flag

        elif direction == 2:  # 南方
            # 如果explored_grids 只包含了一个网格，则先向东探索（设定的规则，后续可优化）
            if len(explored_grids) == 1:
                next_pose = explored_poses[-1].copy()
                next_pose[0] = next_pose[0] + self.exploration_step
                return next_pose, explored_back_flag
            else:
                if explored_back_flag == 0:
                    # 当前是向东探索
                    # 如果当前pose位于待探索区域东侧，说明需要向西探索
                    if current_grid_coord[0] > landmark_exp_area[2]:
                        # 取初始探索点西侧的网格坐标
                        next_pose = explored_poses[0].copy()
                        next_pose[0] = next_pose[0] - self.exploration_step
                        explored_back_flag = 1
                        return next_pose, explored_back_flag
                    else:
                        # 如果没到东侧，则继续向东探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[0] = next_pose[0] + self.exploration_step
                        return next_pose, explored_back_flag
                else:
                    # 当前是向西探索
                    # 如果当前pose位于待探索区域西侧，说明所有区域已经探索完毕，返回探索点None
                    if current_grid_coord[0] < landmark_exp_area[0]:
                        return None, explored_back_flag
                    else:
                        # 如果没到西侧，则继续向西探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[0] = next_pose[0] - self.exploration_step
                        return next_pose, explored_back_flag

        elif direction == 3:  # 西方
            # 如果explored_grids 只包含了一个网格，则先向南探索（设定的规则，后续可优化）
            if len(explored_grids) == 1:
                next_pose = explored_poses[-1].copy()
                next_pose[1] = next_pose[1] + self.exploration_step
                return next_pose, explored_back_flag
            else:
                if explored_back_flag == 0:
                    # 当前是向南探索
                    # 如果当前pose位于待探索区域南侧，说明需要向北探索
                    if current_grid_coord[1] > landmark_exp_area[3]:
                        # 取初始探索点北侧的网格坐标
                        next_pose = explored_poses[0].copy()
                        next_pose[1] = next_pose[1] - self.exploration_step
                        explored_back_flag = 1
                        return next_pose, explored_back_flag
                    else:
                        # 如果没到南侧，则继续向南探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[1] = next_pose[1] + self.exploration_step
                        return next_pose, explored_back_flag
                else:
                    # 当前是向北探索
                    # 如果当前pose位于待探索区域北侧，说明所有区域已经探索完毕，返回探索点None
                    if current_grid_coord[1] < landmark_exp_area[1]:
                        return None, explored_back_flag
                    else:
                        # 如果没到北侧，则继续向北探索
                        next_pose = explored_poses[-1].copy()
                        next_pose[1] = next_pose[1] - self.exploration_step
                        return next_pose, explored_back_flag

    ## 以下为导航相关函数
    def get_navigable_grid(self, obj_id, direction):
        """
        获取物体obj_id对应方向的可导航网格坐标集合
        """
        # 获取物体obj_id的栅格坐标并转换为numpy数组
        obj_grid = self.ObjectSet[obj_id]["grid"]
        coords = np.array(self.grid_id_to_local_coord(obj_grid))

        # 根据方向获取可导航网格集合
        if direction == 0:  # 北
            x_coords = np.unique(coords[:, 0])
            y_values = np.max(coords[:, 1])
            navigable_coords = np.array([[x, y_values + self.navigable_grid_range] for x in x_coords])
        elif direction == 1:  # 东
            y_coords = np.unique(coords[:, 1])
            x_values = np.max(coords[:, 0])
            navigable_coords = np.array([[x_values + self.navigable_grid_range, y] for y in y_coords])
        elif direction == 2:  # 南
            x_coords = np.unique(coords[:, 0])
            y_values = np.min(coords[:, 1])
            navigable_coords = np.array([[x, y_values - self.navigable_grid_range] for x in x_coords])
        elif direction == 3:  # 西
            y_coords = np.unique(coords[:, 1])
            x_values = np.min(coords[:, 0])
            navigable_coords = np.array([[x_values - self.navigable_grid_range, y] for y in y_coords])

        # 过滤掉超出边界的网格
        valid_mask = (navigable_coords >= 0).all(axis=1) & (navigable_coords <= self.map_side_length).all(axis=1)
        navigable_coords = navigable_coords[valid_mask]

        return navigable_coords

    def check_landmark_reached(self, agent_pose, navigable_grid_coords):
        """
        判断agent是否已经到达导航的目的地

        Args:
            agent_pose: agent当前pose，列表，形状为[x, y, z, roll, pitch, yaw]
            navigable_grid_coords: 可导航网格坐标集合，numpy数组，形状为(n, 2)

        Returns:
            flag: 是否到达目的地，布尔值
        """
        agent_grid = self.world_coord_to_grid_id(np.array([agent_pose[:2]]))[0]
        agent_coord = self.grid_id_to_local_coord(np.array([agent_grid]))[0]

         # 由于agent_coord是维度为2的numpy数组，需要使用numpy的array_equal来判断是否相等
        # flag = any(np.array_equal(agent_coord, coord) for coord in navigable_grid_coords)

        # 计算agent_coord与每个navigable_grid_coords之间的曼哈顿距离
        distances = np.sum(np.abs(navigable_grid_coords - agent_coord), axis=1)
        # 判断是否有距离小于等于3的网格
        flag = np.any(distances <= 3)
        return flag


    def get_next_nav_pose(self, agent_pose, navigable_grid_coords):
        """
        获取agent从当前位置到目标网格的最优路径，并根据步长生成路径上的下一个网格
        """

        agent_grid = self.world_coord_to_grid_id(np.array([agent_pose[:2]]))[0]
        agent_grid_coords = self.grid_id_to_local_coord(np.array([agent_grid]))[0]
        # 简单的策略：选择最近的可导航网格
        min_distance = float('inf')
        target_grid_coords = agent_grid_coords

        for grid_coords in navigable_grid_coords:
            distance = np.sum(np.abs(grid_coords - agent_grid_coords))
            if distance < min_distance:
                min_distance = distance
                target_grid_coords = grid_coords

        # 将agent_grid_coords, target_grid_coords中的元素转化为int
        agent_grid_coords = tuple(map(int, agent_grid_coords))
        target_grid_coords = tuple(map(int, target_grid_coords))
        # 利用A*算法计算agent从当前位置到目标网格的最优路径
        path = self.astar_search(agent_grid_coords, target_grid_coords)
       
        if len(path) > self.navigation_step_max:
            next_coord = path[self.navigation_step_max]
        else:
            next_coord = path[-1]
            
        next_pose = agent_pose.copy()
        next_pose[0:2] = [next_coord[0] + self.p_global_origin[0], next_coord[1] + self.p_global_origin[1]]

        advance_distance = np.min([len(path), self.navigation_step_max])
        log_info(f"Advance {advance_distance} meter")
        return next_pose

    def astar_search(self, start, goal):
        """
        在nav_map上使用A*算法计算agent从当前位置到目标网格的最优路径

        Args:
            start: 起点网格坐标，列表或元组，形状为(2,)
            goal: 终点网格坐标，列表或元组，形状为(2,)

        Returns:
            path: 最优路径，列表，每个元素为网格坐标
        """

        def heuristic(a, b):
            # 计算两个点之间的曼哈顿距离作为启发式函数
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def is_valid(coord):
            # 检查坐标是否在地图范围内
            x, y = coord
            return 0 <= x < self.map_side_length and 0 <= y < self.map_side_length

        def is_navigable(coord):
            # 检查坐标是否为可导航区域(假设未知和已探索区域都可以导航)
            x, y = coord
            return self.nav_map[x, y] != 2

        def is_safe(coord):
            # 检查坐标周围是否安全，即不靠近障碍物
            x, y = coord
            x_range = np.arange(max(0, x - self.expected_obstacle_distance), 
                                min(self.map_side_length, x + self.expected_obstacle_distance + 1))
            y_range = np.arange(max(0, y - self.expected_obstacle_distance), 
                                min(self.map_side_length, y + self.expected_obstacle_distance + 1))
            xx, yy = np.meshgrid(x_range, y_range)
            coords = np.vstack([xx.ravel(), yy.ravel()]).T

            if np.any(self.nav_map[coords[:, 0], coords[:, 1]] == 2):
                return False
            return True

        def neighbor_is_valid(coord):
            x, y = coord
            if is_valid((x, y)):
                if is_navigable((x, y)):
                    if is_safe((x, y)):
                        return True
            return False

        # 初始化优先队列和相关数据结构
        open_set = PriorityQueue()
        open_set.put((0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): heuristic(start, goal)}

        while not open_set.empty():
            # 从优先队列中取出具有最低f_score的节点
            _, current = open_set.get()

            # 如果当前节点是目标节点构建路径并返回
            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(tuple(start))
                path.reverse()
                return path

            # 遍历当前节点的邻居
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                # 检查邻居是否有效、可导航且安全
                if neighbor_is_valid(neighbor):
                    tentative_g_score = g_score[current] + heuristic(current, neighbor)
                    # 如果邻居节点未被访问过或发现更短路径，更新路径信息
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        open_set.put((f_score[neighbor], neighbor))

        # 如果没有路径到达目标，找到离目标最近的可导航点的路径
        closest_point = min(g_score.keys(), key=lambda p: heuristic(p, goal))
        path = []
        current = closest_point
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(tuple(start))
        path.reverse()
        return path







