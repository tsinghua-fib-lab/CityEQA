import os

from Utils.common_utils import *
from Utils.arguments import get_args
from Agent.llmAgent import LLMagent



def parse_task(question, llm_agent):

    response = llm_agent.parse_question(question)
    new_re = fix_response(response, ["Object", "Relationship", "Requirement", "Plan"])
    return new_re


if __name__ == "__main__":
    os.chdir('..')
    args = get_args()
    llm_agent = LLMagent(args.model, args.api_key)
    # load dataset
    with open("./test_question.json", "r") as f:
        dataset = json.load(f)
    task = dataset[0]
    question = task["question"]

    response = parse_task(question, llm_agent)

    print(response)
