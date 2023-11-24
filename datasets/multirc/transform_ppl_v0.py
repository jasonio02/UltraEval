import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"{data['passage']}\nQuestion: {data['question'][0]}\nClaim: {data['question'][1]}\nIs it true? "
    correct_answer = [
        key for key, value in data["target_scores"].items() if value == 1
    ][0].strip()

    return {"input": text, "output": correct_answer, "processed_output": correct_answer}
