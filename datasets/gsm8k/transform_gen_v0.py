import random

from UltraEval.tasks.postprocess import GSM8KPost
import re

prompt_tuning = """Your job is to think through every question step by step and give an answer. Then, give your final answer followed by “### ”. Here are some examples: 

Question1: Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?

Answer1: Half of the number of Randy's mango trees is 60/2 = <<60/2=30>>30 trees. So Randy has 30 - 5 = <<30-5=25>>25 coconut trees. Therefore, Randy has 60 + 25 = <<60+25=85>>85 treeson his farm. #### 85 """


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = "Question: {data['question']}\nAnswer: "
    correct_answer = data["answer"]

    regex_expressions = re.findall(r'\b\d+\s*[\+\-\*/]\s*\d+(\s*[\+\-\*/]\s*\d+)*', data['question'])
    if regex_expressions:
        text += f"'\nMath Expressions:{','.join(regex_expressions)}\n"

    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
