
import warnings
import numpy as np

from Agent.airsimAgent import AirsimAgent
from Agent.llmAgent import LLMagent
from Agent.actor import Collector
from Utils.common_utils import *
from Utils.arguments import get_args

warnings.filterwarnings("ignore")
# 初始化日志记录器
logger = logging.getLogger("global_logger")
logger.setLevel(logging.INFO)  # 设置日志级别


def main():
    args = get_args()
    airsim_agent = AirsimAgent()
    llm_agent = LLMagent(args.model, args.api_key)

    # load dataset
    dataset_file_path = args.output_directory / ("LLM" + "/answer_s1.json")
    dataset = json.load(dataset_file_path.open("r"))
    data_len = len(dataset)

    output_path = args.output_directory / "LLM"
    answer_file_path = output_path / "answer_s2.json"
    log_folder_path = output_path / "history"

    results = []
    first_idx = 0

    # if first_idx != 0:
    #     answer_file_path = output_path / f"answer_s2_{first_idx}.json"

    is_save_img = False

    maxStep = 10

    args.collector_no_move = False
    collector_agent = Collector(args, llm_agent)

    for idx in range(first_idx, data_len):
        # 测试模式下，只运行一次
        # if args.dry_run and idx >= 1:
        #     break

        # ---------- Initial Settings ---------- #
        task_item = dataset[idx]
        question_id = task_item["question_id"]
        stage_1_pose = task_item["stage_1_pose"]
        stage_1_step = task_item["stage_1_step"]
        ground_truth_answer = task_item["ground_truth_answer"]
        task = task_item["collect_task"]
        collector_agent.reset(task)
        print(f"Now the task id: {question_id}")

        folder_path = os.path.join(log_folder_path, question_id)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 初始化log配置
        log_file = os.path.join(folder_path, 'execution_s2.log')
        change_log_file(log_file)
        # 记录任务执行过程中的重要信息
        log_info(f'Question ID: {question_id}')
        log_info(f'Stage_1_Pose: {stage_1_pose}')
        log_info(f'Stage_1_Step: {stage_1_step}')
        log_info(f'ground_truth_answer: {ground_truth_answer}')
        log_info(f'Task: {task}')

        airsim_agent.setVehiclePose(stage_1_pose)
        # 时停1s非常重要，确保初始位置已经传进去了
        time.sleep(1)

        step = 1
        # 需要记录的变量，stage_2是收集阶段
        stage_2_result = []

        # 超过5步不动就退出
        flag_stop_count = 0

        # ---------- Start ---------- #
        while step <= maxStep:
            log_info(f'\n*** Step {step} ***')

            # get and save new observation
            pose = airsim_agent.get_current_state()
            log_info(f'pose {pose}')
            agent_pose = pose.copy()

            log_info('Get observation')
            img_BGR = airsim_agent.get_rgb_image()

            if is_save_img:
                log_info('Save observation')
                save_observation_s2(img_BGR, pose, folder_path, step)

            log_info('Execute Collection')
            next_pose, answer, action = collector_agent.step(img_BGR, agent_pose)

            if np.array_equal(next_pose, pose):
                flag_stop_count += 1
                if flag_stop_count >= 4:
                    log_info('No further movement')
                    break
            else:
                flag_stop_count = 0
                print(f"count={flag_stop_count}")

            # if step in answer_step:
            step_result = {"step": step, "pose": pose.tolist(), "answer": answer, "action": action}
            stage_2_result.append(step_result)
            
            log_info('Move to new pose')
            airsim_agent.setVehiclePose(next_pose)
            # 时停1s非常重要，确保初始位置已经传进去了
            time.sleep(1)
            step += 1

        task_result = {
            "question_id": question_id,
            "question": task_item["question"],
            "meta_question": task["requirement"],
            "ground_truth_answer": task_item["ground_truth_answer"],
            "initial_pose": task_item["initial_pose"],
            "target_pose": task_item["target_pose"],
            "stage_1_pose": task_item["stage_1_pose"],
            "stage_1_step": task_item["stage_1_step"],
            "stage_2_result": stage_2_result
        }
        log_info(f'\n Task Result: {task_result}')

    # store results
        results.append(task_result)
        json.dump(results, answer_file_path.open("w"), indent=4)


if __name__ == "__main__":

    main()
