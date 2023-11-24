import random

from UltraEval.tasks.postprocess import TheoremQAPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"\nQuestion: {data['question']}\nAnswer: "
    correct_answer = str(data["answer"][1])
    tqap = TheoremQAPost()
    _, processed_correct_answer = tqap([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
