import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def get_harmful_instructions():
    dataset = pd.read_csv("./data/harmful_instructions.csv")
    instructions = dataset["goal"].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


print(get_harmful_instructions())
