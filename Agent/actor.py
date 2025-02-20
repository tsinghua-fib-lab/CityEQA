import numpy as np
from Utils.common_utils import *
from Agent.llmAgent import LLMagent
from Utils.arguments import get_args

direction_map = {
        'north': 0,
        'east': 1, 
        'south': 2,
        'west': 3
        }


class Explorer:
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent
        # 任务信息
        self.task = None  
        # 参照物名称
        self.landmark = None
        # 参照物id
        self.landmark_id = None
        # 参照物与目标的空间关系
        self.ref = None
        # 目标名称
        self.target = None
        # 目标特征
        self.characteristic = None
        # 目标类别
        self.target_class = None
        # 目标相对于参照物的方向
        self.direction = None

        # 环顾周围标志(0: 前视，1: 左视，2: 后视，3: 右视)
        self.check_around_flag = None
        # 已探索网格列表
        self.explored_poses = None
        # 折返探索标志
        self.explored_back_flag = None

    def reset(self, task, objset):
        self.task = task
        self.target = task["target"]
        self.characteristic = task["characteristic"]
        self.landmark = task["landmark"]
        self.target_class = objset[self.target]["type"]
        self.check_around_flag = 0
        self.explored_poses = []
        self.explored_back_flag = 0

        self.direction = direction_map[self.task["direction"]]

        # 判断object[self.landmark]["id"]是否存在
        if objset[self.landmark]["state"] == "unknown":
            log_info("In exploration, landmark does not found")
            return False
        else:
            self.landmark_id = objset[self.landmark]["id"]
            return True

    def check_target_appeared_in_cogmap(self, cogmap_agent):
        # 判断target是否在cogmap中
        target_id = cogmap_agent.check_target_appeared(self.target_class, self.landmark_id, self.direction)
        if target_id is None:
            return None
        else:
            return target_id

    def check_target_appeared_in_rgb(self, img_BGR):
        # 判断target是否在rgb图像中
        flag_target_appeared_in_rgb = self.llm_agent.obj_in_rgb(img_BGR, self.target, self.characteristic)

        return flag_target_appeared_in_rgb == 'True'

    def get_next_exploration_pose(self, cogmap_agent):
        # 获取下一个探索点
        next_pose, self.explored_back_flag = cogmap_agent.get_next_exploration_pose(self.explored_poses, self.landmark_id, self.direction, self.explored_back_flag)
        if next_pose is not None:
            return next_pose
        else:
            return None

    def step(self, img_BGR, cogmap_agent, objset, agent_pose):
        # 更新landmark_id
        self.landmark_id = objset[self.landmark]["id"]

        if self.target_class in cogmap_agent.landmark_list:
            # 检查target是否在cogmap中
            target_id = self.check_target_appeared_in_cogmap(cogmap_agent)
            if target_id is not None:
                # 将target_id加入到objset中
                objset[self.target]["id"] = target_id
                objset[self.target]["state"] = "known"
                log_info(f"Exploration completed: The target has already appeared in cognitive map")
                return objset, agent_pose, True
            else:
                log_info(f"The target has not appeared in cognitive map")
        else:
            # 检查当前视图是否存在目标
            flag_target_appeared_in_rgb = self.check_target_appeared_in_rgb(img_BGR)
            # flag_target_appeared_in_rgb = False
            if flag_target_appeared_in_rgb:
                log_info(f"Exploration completed: The target has already appeared in current view")
                return objset, agent_pose, True
            else:
                log_info(f"The target has not appeared in current view")

        # 如果当前cogmap或视图不存在目标，则选择环顾或移动到下一个探索点
        if self.check_around_flag < 3:
            self.check_around_flag += 1
            agent_pose[5] += 90
            log_info(f"Look around ({self.check_around_flag}/4)")
            return objset, agent_pose, False
        else:
            self.check_around_flag = 0
            self.explored_poses.append(agent_pose.copy())  # 使用copy()避免引用问题
            # 获取下一个探索点
            next_agent_pose = self.get_next_exploration_pose(cogmap_agent)
            # 如果下一个探索点为None，则返回True(表示探索任务完成), 否则返回False
            if next_agent_pose is None:
                log_info("Exploration completed: All exploration poses have been explored")
                return objset, agent_pose, True
            else:
                log_info(f"Look around (4/4)")
                log_info("Move to the next exploration point")
                return objset, next_agent_pose, False


class Navigator:
    def __init__(self):
        # 任务信息
        self.task = None  
        # 参照物名称
        self.landmark = None
        # 参照物id
        self.landmark_id = None
        # 导航到参照物的方向
        self.direction = None

    def reset(self, task, objset):
        self.task = task
        self.landmark = task["landmark"]
        self.direction = direction_map[self.task["direction"]]

        # 判断object[self.landmark]["id"]是否存在
        if objset[self.landmark]["state"] == "unknown":
            log_info("In Navigation, landmark does not found")
            return False
        else:
            self.landmark_id = objset[self.landmark]["id"]

            return True

    def step(self, cogmap_agent, agent_pose, objset):
        # 更新landmark_id
        self.landmark_id = objset[self.landmark]["id"]
        # 获取参照物对应方向的可导航网格集合
        navigable_grid_coords = cogmap_agent.get_navigable_grid(self.landmark_id, self.direction)

        # 判断agent是否已经在导航的目的地
        flag_landmark_reached = cogmap_agent.check_landmark_reached(agent_pose, navigable_grid_coords)

        if flag_landmark_reached:
            log_info("Navigation completed. Agent has already arrived at the landmark")
            return agent_pose, True

        # 获取下一位置
        next_pose = cogmap_agent.get_next_nav_pose(agent_pose, navigable_grid_coords)
        return next_pose, False
    

class Collector:
    def __init__(self, args, llm_agent):
        self.llm_agent = llm_agent
        # 任务信息
        self.task = None  
        # 目标名称
        self.target = None
        # 目标特征
        self.characteristic = None
        # 目标指令
        self.requirement = None
        # 历史信息
        self.history_info = None
        # 收集信息步数
        self.collect_step = None
        # # 最大收集信息步数
        # self.max_collect_step = 20
        # 是否不移动
        self.no_move = args.collector_no_move

        self.answer = None

    def reset(self, task, pre_answer=None):
        self.task = task
        self.target = task["target"]
        self.characteristic = task["characteristic"]
        self.requirement = task["requirement"]
        self.collect_step = 0
        if pre_answer is not None:
            self.history_info = [f"step 0: Answer: {pre_answer}"]
        else:
            self.history_info = []

        self.answer = None

    def step(self, img_BGR, agent_pose):

        # next_pose = agent_pose + 1
        # return next_pose, None, True

        if self.no_move:
            return agent_pose, None, True

        # 调整视角，收集信息
        self.collect_step += 1
        response = self.llm_agent.collect_move(img_BGR, self.requirement, self.answer)

        action = response["Action"]
        answer = response["Answer"]
        reason = response["Reason"]
        log_info(f'Action: {action}, Answer: {answer}')
        log_info(f'Reason: {reason}')

        next_pose = self.get_next_pose(agent_pose, action)
        history_info = f"step {self.collect_step}: Action: {action}, Answer: {answer}"
        self.history_info.append(history_info)
        self.answer = answer

        return next_pose, answer, action

    def get_next_pose(self, agent_pose, action):
        # 获取下一个位置
        distance = 2
        add_yaw_angle = 10

        pos = agent_pose[:3]
        ori = agent_pose[3:]
        yaw = np.deg2rad(ori[2])

        if action == "MoveForward":
            pos[0] = pos[0] + distance * np.sin(yaw)
            pos[1] = pos[1] + distance * np.cos(yaw)
            # print(f"Action: move forward {distance} meters\n\n")
        elif action == "MoveBack":
            pos[0] = pos[0] - distance * np.sin(yaw)
            pos[1] = pos[1] - distance * np.cos(yaw)
            # print(f"Action: move back {distance} meters\n\n")
        elif action == "MoveLeft":
            pos[0] = pos[0] - distance * np.cos(yaw)
            pos[1] = pos[1] + distance * np.sin(yaw)
            # print(f"Action: move left {distance} meters\n\n")
        elif action == "MoveRight":
            pos[0] = pos[0] + distance * np.cos(yaw)
            pos[1] = pos[1] - distance * np.sin(yaw)
            # print(f"Action: move right {distance} meters\n\n")
        elif action == "TurnLeft":
            ori[2] = ori[2] - add_yaw_angle
            # print(f"Action: turn left {add_yaw_angle} degrees\n\n")
        elif action == "TurnRight":
            ori[2] = ori[2] + add_yaw_angle
            # print(f"Action: turn right {add_yaw_angle} degrees\n\n")
        elif action == "MoveUp":
            pos[2] = pos[2] + distance/2
            # print(f"Action: move up {distance/2} meters\n\n")
        elif action == "MoveDown":
            pos[2] = pos[2] - distance/2
            # print(f"Action: move down {distance/2} meters\n\n")
        else:
            log_info(f"Action: Unknown action {action}, keep still")

        next_pose = np.concatenate((pos, ori))

        return next_pose

    def get_answer_no_move(self, img_BGR):
        answer = self.llm_agent.collect_QA(img_BGR, self.requirement)
        return answer
