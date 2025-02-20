import os
import csv
import time
import cv2
import matplotlib.pyplot as plt
import json
import base64
import math
import logging
import re

logger = logging.getLogger("global_logger")


class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        # 检查是否是列表，并且是特定的字段（如 target_pose）
        if isinstance(obj, list) and any(isinstance(item, (int, float)) for item in obj):
            return json.dumps(obj, separators=(',', ':'))  # 使用紧凑格式
        return super().encode(obj)


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        # 使用自定义的 JSON 编码器
        json.dump(data, file, indent=4, cls=CustomJSONEncoder)


def write_to_csv(stage, step, str_time, pos, ori, folder_path):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, 'record.csv')

    # 检查文件是否存在，如果不存在则创建文件并写入表头
    file_exists = os.path.isfile(file_path)

    # 打开文件，如果文件不存在则创建，并设置newline=''以避免写入额外的空行
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 如果文件是新创建的，写入表头
        if not file_exists:
            csvwriter.writerow(['Stage','Step', 'Time', 'X', 'Y', 'Z', 'Pitch', 'Roll', 'Yaw'])

        # 写入一行数据
        csvwriter.writerow([stage] + [step] + [str_time] + list(pos) + list(ori))


def save_observation_s1(img_rgb, img_depth, pose, folder_path, step):

    pos, ori = pose[:3], pose[3:]
    rgb_save_name = os.path.join(folder_path, f'{step}_s1_rgb.png')
    d_save_name = os.path.join(folder_path, f'{step}_s1_d.png')
    # 保存为.png格式的图像文件
    cv2.imwrite(rgb_save_name, img_rgb)
    cv2.imwrite(d_save_name, img_depth)

    str_time = time.strftime("%Y%M%d%H%M%S", time.localtime(time.time()))
    write_to_csv("s1", step, str_time, pos, ori, folder_path)
    # print(f"Step {step} successfully record")


def save_observation_s2(img_rgb, pose, folder_path, step):

    pos, ori = pose[:3], pose[3:]
    rgb_save_name = os.path.join(folder_path, f'{step}_s2_rgb.png')
    # 保存为.png格式的图像文件
    cv2.imwrite(rgb_save_name, img_rgb)

    str_time = time.strftime("%Y%M%d%H%M%S", time.localtime(time.time()))
    write_to_csv("s2", step, str_time, pos, ori, folder_path)
    # print(f"Step {step} successfully record")


def display_image_bgr(image_bgr):
    # 将BGR图像转换为RGB格式
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # 不显示坐标轴
    plt.show(block=False)  # 非阻塞显示
    plt.pause(0.5)  # 暂停1秒


def fix_response(response, required_keys):
    """
    修复响应，确保其符合指定的字典格式。

    :param response: dict 或 str, 原始响应
    :param required_keys: list, 必需的字段列表
    :return: dict, 修复后的响应
    """
    # 如果输出是字典
    if isinstance(response, dict):
        for key in required_keys:
            if key not in response:
                response[key] = 'None'
        return response

    # 如果输出是字符串
    elif isinstance(response, str):
        # 尝试提取 '{' 和 '}' 之间的内容
        match = re.search(r'\{(.*)\}', response, re.DOTALL)
        if match:
            json_like_content = "{" + match.group(1) + "}"
            try:
                # 尝试将提取的内容解析为 JSON
                parsed_output = json.loads(json_like_content)
                if isinstance(parsed_output, dict):
                    # 补全缺失的字段为 None
                    for key in required_keys:
                        if key not in parsed_output:
                            parsed_output[key] = 'None'
                    return parsed_output
            except json.JSONDecodeError:
                # 如果 JSON 解析失败，继续尝试从文本中提取信息
                pass

        # 如果 JSON 解析失败，使用正则表达式逐项提取字段
        extracted_data = {}
        for key in required_keys:
            # 使用正则表达式提取每个字段的信息
            match = re.search(rf'"{key}"\s*:\s*"(.*?)"', response)
            if match:
                extracted_data[key] = match.group(1)
            else:
                extracted_data[key] = 'None'

        return extracted_data

    # 如果输出既不是字典也不是字符串，则将required_keys中每一个字段的值赋为"None"并返回
    else:
        none_data = {}
        for key in required_keys:
            none_data[key] = 'None'
        return none_data


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()


# 动态改变日志文件存储位置
def change_log_file(new_log_file):
    # 移除当前的文件处理器
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    # 创建新的文件处理器
    new_handler = logging.FileHandler(new_log_file, mode="w")
    new_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    new_handler.setFormatter(formatter)  # 使用相同的格式器
    logger.addHandler(new_handler)


def log_info(info):
    logger.info(info)
    # print(info)
