import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = (
        "Passage: "
        + data["passage"]
        + "\nDoes the pronoun # "
        + data["question"][1]
        + " # refer to * "
        + data["question"][0]
        + " *?\nAnswer: "
    )
    correct_answer = [
        key for key, value in data["target_scores"].items() if value == 1
    ][0].strip()

    return {"input": text, "output": correct_answer, "processed_output": correct_answer}
