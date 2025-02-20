from Agent.actor import Explorer, Navigator, Collector
from Utils.common_utils import *

class Manager:
    def __init__(self, args, llm_agent):

        # 感知计划
        self.plan = None
        # 问题中的物体集
        self.object_set = None
        # 原始指令
        self.instruction = None
       # 原始指令中包含的问题
        self.question = None
        # 是否完成stage_1
        self.stage_1_done = False
        # 答案
        self.answer_stage_1 = None
        self.answer_stage_2 = None
        # 当前任务id
        self.current_task_id = 0
        # 当前任务
        self.current_task = None
        # 当前任务类型
        self.current_task_type = None

        self.llm_agent = llm_agent
        # 初始化
        self.explorer = Explorer(llm_agent)
        self.navigator = Navigator()
        self.collector = Collector(args, llm_agent)

    def reset(self, instruction, response):

        self.object_set = response['Object']
        self.question = response['Requirement']
        self.plan = response['Plan']
        self.instruction = instruction
        self.answer_stage_1 = None
        self.answer_stage_2 = None
        self.stage_1_done = False

        self.current_task_id = 0
        self.current_task = self.plan[self.current_task_id]
        self.current_task_type = self.current_task['type']

        self.object_set['drone']['id'] = "drone"

        while True:
            if self.current_task_type == "Exploration":
                flag = self.explorer.reset(self.current_task, self.object_set)
            elif self.current_task_type == "Navigation":
                flag = self.navigator.reset(self.current_task, self.object_set)
            elif self.current_task_type == "Collection":
                self.collector.reset(self.current_task)
                flag = True
                self.stage_1_done = True
                
            if flag:
                log_info(f"{self.current_task_type} task initialization success")
                break
            else:
                log_info(f"{self.current_task_type} task initialization fail")
                # 如果还有任务，则切换到下一个任务
                if self.current_task_id < len(self.plan) - 1:
                    self.current_task_id += 1

                    self.current_task = self.plan[self.current_task_id]
                    self.current_task_type = self.current_task['type']
                else:
                    log_info("All tasks initialization fail")
                    return False
        return True

    def update_object_id(self, cogmap_agent):
        # 根据cogmap_agent中的物体融合记录，更新self.object_set中的物体信息
        # 取cogmap_agent中的obj_merge_id
        obj_merge_id = cogmap_agent.obj_merge_id
        # 如果obj_merge_id不为空，则更新self.object_set中的物体信息
        if obj_merge_id:
            flag = True
            while flag:
                flag = False
                # 取obj_merge_id中的所有的键
                update_keys = list(obj_merge_id.keys())
                # 遍历self.object_set中的键和值
                for obj_name, obj in self.object_set.items():
                    # 如果物体id存在
                    if 'id' in obj:
                        # 如果物体id在obj_merge_id中，则更新物体id
                        if obj['id'] in update_keys:
                            self.object_set[obj_name]['id'] = obj_merge_id[obj['id']]
                            flag = True

    def get_answer_stage_1(self, img_BGR):
        answer = self.collector.get_answer_no_move(img_BGR)
        self.answer_stage_1 = answer
        return answer

    def save_cog_map(self, save_path, cogmap_agent):
        # 判断landmark是否已经有id了
        if self.object_set['building_1']["state"] == "unknown":
            landmark_id = None
        else:
            landmark_id = self.object_set['building_1']["id"]
        cogmap_agent.save_map_landmark(save_path, landmark_id)

    def step(self, img_BGR, agent_pose, cogmap_agent):
        
        # 根据cogmap_agent中的物体融合记录，更新self.object_set中的物体信息
        self.update_object_id(cogmap_agent)
        next_pose = None
        plan_done = False
        task_done = False

        while True:
            # 根据当前任务类型，执行任务
            if self.current_task_type == "Exploration":
                # 执行探索任务
                log_info("Current task: Exploration")
                self.object_set, next_pose, task_done = self.explorer.step(img_BGR, cogmap_agent, self.object_set, agent_pose)
                
            elif self.current_task_type == "Navigation":
                log_info("Current task: Navigation")
                next_pose, task_done = self.navigator.step(cogmap_agent, agent_pose, self.object_set)

            elif self.current_task_type == "Collection":
                # log_info("Current task: Collection")
                # next_pose, answer, task_done = self.collector.step(img_BGR, agent_pose)
                # if task_done:
                #     self.answer_stage_2 = answer
                task_done = True

            # 如果任务完成
            if task_done:
                while True:
                    # 如果还有任务，则切换到下一个任务
                    if self.current_task_id < len(self.plan) - 1:
                        self.current_task_id += 1
                        self.current_task = self.plan[self.current_task_id]
                        self.current_task_type = self.current_task['type']
                        log_info(f"Current task done, switch to next task: {self.current_task_type}")
                        task_done = False

                        # 如果当前任务是探索任务，则重置explorer
                        if self.current_task_type == "Exploration":
                            reset_flag = self.explorer.reset(self.current_task, self.object_set)

                        elif self.current_task_type == "Navigation":
                            reset_flag = self.navigator.reset(self.current_task, self.object_set)
    
                        elif self.current_task_type == "Collection":
                            self.collector.reset(self.current_task)
                            reset_flag = True
                            self.stage_1_done = True
                        
                        if reset_flag:
                            log_info(f"{self.current_task_type} task initialization success")
                            break
                        else:
                            log_info(f"{self.current_task_type} task initialization fail")

                    else:
                        next_pose = None
                        plan_done = True
                        return next_pose, plan_done
                    
            else:
                break
            
        return next_pose, plan_done

    def get_collect_task(self):
        for task in self.plan:
            if task["type"] == "Collection":
                return task
        return None


if __name__ == "__main__":
    pass

