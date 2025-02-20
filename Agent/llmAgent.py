import numpy as np
from openai import OpenAI
import os

from Utils.common_utils import *
from Utils.arguments import get_args
import cv2
import base64


class LLMagent:
    """
    Large Model Agent
    """
    def __init__(self, model, api_key):
        """
        :param model: LM model
        :param api_key: api key corresponding to the LM model
        """
        self.model = model
        self.model_class = model.split('-')[0]
        if self.model_class == 'qwen':
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url=""
            )
        elif self.model_class == 'gpt':
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url="",
            )
        else:
            raise ValueError(f"Unknown evaluation model type {self.eval_model}")

    def obj_in_rgb(self, img_BGR, target, characteristic):
        # 判断target是否在rgb图像中

        # 将图像转换为PNG格式的字节数据
        _, png_bytes = cv2.imencode('.png', img_BGR)
        base64_image = base64.b64encode(png_bytes).decode("utf-8")

        system_message = load_text(f"./prompts/obj_in_rgb.txt")
        # 如果characteristic为空，则不添加到user_message中
        if characteristic:
            user_message = f"The target is {target}, the characteristic is {characteristic}."
        else:
            user_message = f"The target is {target}."

        try:
            # time.sleep(5)
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {
                                "type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                             },
                        ],
                    }
                ],
            )
            response = completion.choices[0].message.content
        except:
            log_info('Failed: LLM cannot response, cannot find the target in the image')
            response = False

        return response

    def get_blind_answer(self, question):

        system_message = load_text(f"./prompts/blind_answer.txt")
        user_message = f"Question: {question}"
        try:    
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            )
            response = completion.choices[0].message.content
        except:
            log_info('Failed: LLM cannot response, cannot get the blind answer')
            response = False
        return response

    def collect_move(self, img_BGR, question, history_info):
        system_message = load_text(f"./prompts/collector_VLA.txt")
        user_message = f"Question: {question}\n"
        user_message += f"Reference answer: {history_info}"

        # 将图像转换为PNG格式的字节数据# 将图像转换为PNG格式的字节数据
        _, png_bytes = cv2.imencode('.png', img_BGR)
        base64_image = base64.b64encode(png_bytes).decode("utf-8")

        for try_time in range(4):
            try:
                completion = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user",
                         "content": [
                             {"type": "text", "text": user_message},
                             {
                                 "type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                             },
                         ],
                         }
                    ],
                )
                response = completion.choices[0].message.content
                break
            except:
                log_info(f'Try {try_time}, LLM cannot response for VLA')
                response = False

        vaild_response = fix_response(response, ["Action", "Answer", "Reason"])
        return vaild_response

    def collect_QA(self, img_BGR, question):
        system_message = load_text(f"./prompts/collector_QA.txt")
        user_message = f"Question: {question}"

        # 将图像转换为PNG格式的字节数据# 将图像转换为PNG格式的字节数据
        _, png_bytes = cv2.imencode('.png', img_BGR)
        base64_image = base64.b64encode(png_bytes).decode("utf-8")

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": user_message},
                         {
                             "type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                         },
                     ],
                     }
                ],
            )
            response = completion.choices[0].message.content
        except:
            log_info('Failed: LLM cannot response, cannot get the collector QA')
            response = False
        return response

    def parse_question(self, question):
        system_message = load_text(f"./prompts/parser.txt")
        user_message = f"Instruction: {question}"
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                # response_format={type: "json_object"},
            )
            response = completion.choices[0].message.content
        except:
            log_info('Failed: LLM cannot response, cannot parse the question')
            response = False
        return response

    def get_score(self, question, answer, prediction):

        system_message = load_text(f"./Evaluation/score.txt")
        user_message = f"Question: {question}\n"
        user_message += f"Answer: {answer}\n"
        user_message += f"Response: {prediction}\n"

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            response = completion.choices[0].message.content
        except:
            print('Failed: LLM cannot response')
            response = False

        vaild_response = fix_response(response, ["mark"])
        mark = vaild_response["mark"]
        return mark





