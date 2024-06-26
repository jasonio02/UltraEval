import random
import re

from UltraEval.tasks.postprocess import GSM8KPost

def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    question = f"Question:\n{data['question']}?\n"
    answer_prompt = f"Answer:\n"
    text = question + answer_prompt
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


