import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    question = f"问题：\n以下两句话是什么关系？\n"
    context = f"第一句话：{data['passage'][0]}\n第二句话：{data['passage'][1]}\n"
    instruction = f"要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n"
    options = "选项：\n"
    for idx, item in enumerate(data["target_scores"].keys()):
        options += f"({chr(65 + idx)}) {item}\n"
    answer_prompt = f"答案：\n"
    text = question + context + instruction + options + answer_prompt
    index_of_correct_answer = list(data["target_scores"].values()).index(1)
    processed_correct_answer = correct_answer = f"({chr(65 + index_of_correct_answer)})"

    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
