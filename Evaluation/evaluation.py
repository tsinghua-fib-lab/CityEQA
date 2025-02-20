import json
import os
import csv
from Agent.llmAgent import LLMagent


def write_to_csv(file_path, results):

    # 检查文件是否存在，如果不存在则创建文件并写入表头
    file_exists = os.path.isfile(file_path)

    # 打开文件，如果文件不存在则创建，并设置newline=''以避免写入额外的空行
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 如果文件是新创建的，写入表头
        if not file_exists:
            csvwriter.writerow(['question_id', 's1_step',
                                '1_distance', '1_score', '1_action',
                                '2_distance', '2_score', '2_action',
                                '3_distance', '3_score', '3_action',
                                '4_distance', '4_score', '4_action',
                                '5_distance', '5_score', '5_action',
                                '6_distance', '6_score', '6_action',
                                '7_distance', '7_score', '7_action',
                                '8_distance', '8_score', '8_action',
                                '9_distance', '9_score', '9_action',
                                '10_distance', '10_score', '10_action',
                                ])
        # 写入一行数据
        csvwriter.writerow(results)


def get_distance(pose1, pose2):
    x1 = pose1[0]
    y1 = pose1[1]

    x2 = pose2[0]
    y2 = pose2[1]

    distance = abs(x1 - x2) + abs(y1 - y2)

    return distance


def main():
    model = "gpt-4"
    key = ''
    llm_scorer = LLMagent(model, key)

    # load dataset
    dataset_path = './Results/LLM/answer_s2.json'
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
    data_len = len(dataset)

    file_path = "./evaluation/results/pma_result.csv"

    first_idx = 0
    for idx in range(first_idx, data_len):

        task_item = dataset[idx]
        question_id = task_item["question_id"]
        print(question_id)
        question = task_item["meta_question"]
        ground_truth_answer = task_item["ground_truth_answer"]
        target_pose = task_item["target_pose"]
        s1_step = task_item["stage_1_step"]
        results = [question_id, s1_step]

        stage_2_result = task_item["stage_2_result"]
        step_num = len(stage_2_result)

        answer_score = {}
        for s_id in range(10):
            if s_id < step_num:
                s_pose = stage_2_result[s_id]["pose"]
                s_answer = stage_2_result[s_id]["answer"]
                s_action = stage_2_result[s_id]["action"]

                s_distance = get_distance(s_pose, target_pose)
                if s_answer in answer_score:
                    s_score = answer_score[s_answer]
                else:
                    s_score = llm_scorer.get_score(question, ground_truth_answer, s_answer)
                    # s_score = 1
                    answer_score[s_answer] = s_score
                s_r = [s_distance, s_score, s_action]

            results += s_r

        write_to_csv(file_path, results)


if __name__ == "__main__":
    os.chdir('..')
    main()