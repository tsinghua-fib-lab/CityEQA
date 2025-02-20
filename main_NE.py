

import warnings

from Agent.airsimAgent import AirsimAgent
from Agent.llmAgent import LLMagent
from Agent.manager import Manager
from Agent.parser import *
from Agent.mapAgent import CognitiveMap
from GroundSAM.groundSam import GroundSAM
from Utils.common_utils import *
from Utils.arguments import get_args

warnings.filterwarnings("ignore")
# 初始化日志记录器
logger = logging.getLogger("global_logger")
logger.setLevel(logging.INFO)  # 设置日志级别


def main():
    args = get_args()
    airsim_agent = AirsimAgent()
    groundsam_agent = GroundSAM(args)
    llm_agent = LLMagent(args.model, args.api_key)
    cogmap_agent = CognitiveMap(args)
    manager_agent = Manager(args, llm_agent)

    # load dataset

    dataset = json.load(args.dataset.open("r"))
    data_len = len(dataset)
    print(data_len)
    output_path = args.output_directory / "LLM"
    file_path = output_path / "answer_s1.json"
    log_folder_path = output_path / "history"

    results = []
    first_idx = 0
    maxStep = 50
    is_save_img = False

    for idx in range(first_idx, data_len):
        # 测试模式下，只运行一次
        # if args.dry_run and idx >= 2:
        #     break
        print(f"id:{idx}, all: {data_len}")
        # ---------- Initial Settings ---------- #
        task_item = dataset[idx]
        question_id = task_item["question_id"]
        question = task_item["question"]
        initial_pose = task_item["initial_pose"]
        ground_truth_answer = task_item["answer"]

        folder_path = os.path.join(log_folder_path, question_id)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 初始化log配置
        log_file = os.path.join(folder_path, 'execution_s1.log')
        change_log_file(log_file)

        # 记录任务执行过程中的重要信息
        log_info(f'Question ID: {question_id}')
        log_info(f'Question: {question}')
        log_info(f'Initial Pose: {initial_pose}')
        log_info(f'Ground Truth Answer: {ground_truth_answer}')

        airsim_agent.setVehiclePose(initial_pose)
        # 时停1s非常重要，确保初始位置已经传进去了
        time.sleep(1)

        step = 1

        # 分解问题
        response = parse_task(question, llm_agent)
        # response = parse_test()
        log_info(f'Question Parse Result: {response}')

        # 初始化认知地图
        cogmap_agent.reset(initial_pose)
        # 初始化manager
        reset_success = manager_agent.reset(question, response)
        if not reset_success:
            log_info(f'Manager initialization fail')
            continue
        log_info(f'Manager initialization success')

        # 需要记录的变量，stage_1是探索+导航阶段
        stage_1_pose = initial_pose.copy()
        stage_1_step = 1
        stage_1_answer = None

        # ---------- Start ---------- #
        while step <= maxStep:
            log_info(f'\n*** Step {step} ***')

            # get and save new observation
            pose = airsim_agent.get_current_state()
            log_info(f'pose {pose}')
            agent_pose = pose.copy()

            log_info('Get observation')
            img_BGR, img_depth = airsim_agent.get_rgbd_image()

            log_info('Get grounding and SAM')
            boxes_filt, pred_labels, pred_masks = groundsam_agent.get_groundsam(img_BGR)

            log_info('Update Cognitive Map')
            masks_squeezed = pred_masks.cpu().numpy()[:, 0, :, :]
            cogmap_agent.process_new_obs(step, img_depth, agent_pose, masks_squeezed, pred_labels)

            log_info('Execute Plan')
            next_pose, plan_done = manager_agent.step(img_BGR, agent_pose, cogmap_agent)

            if is_save_img:
                log_info('Save observation')
                save_observation_s1(img_BGR, img_depth, pose, folder_path, step)

                log_info('Save grounding and SAM')
                groundsam_agent.save_result(img_BGR, boxes_filt, pred_labels, pred_masks, folder_path, step)

                log_info('Save Cognitive Map')
                save_name = os.path.join(folder_path, f"map_{step}.jpg")
                manager_agent.save_cog_map(save_name, cogmap_agent)

            if plan_done:
                break

            log_info('Move to new pose')
            airsim_agent.setVehiclePose(next_pose)
            # 时停1s非常重要，确保初始位置已经传进去了
            time.sleep(1)
            step += 1

        stage_1_pose = pose.tolist()
        stage_1_step = step
        # stage_1_answer = manager_agent.get_answer_stage_1(img_BGR)
        collect_task = manager_agent.get_collect_task()
        log_info(f'Stage 1 Done')
        log_info(f'stage_1_answer: {stage_1_answer}')

        task_result = {
            "question_id": question_id,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "initial_pose": initial_pose,
            "target_pose": task_item["target_pose"],
            "stage_1_pose": stage_1_pose,
            "stage_1_step": stage_1_step,
            "stage_1_answer": stage_1_answer,
            "collect_task": collect_task
        }
        log_info(f'Task Result: {task_result} \n\n')

    # store results
        results.append(task_result)
        json.dump(results, file_path.open("w"), indent=2)


if __name__ == "__main__":

    main()
