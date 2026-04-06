import random
from datasets import load_dataset
from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

dataset = load_dataset("boolq")

train_data = dataset["train"]
val_data = dataset["validation"]

def format_example(example):
    return f"Question: {example['question']}\nPassage: {example['passage']}\nAnswer: {'yes' if example['answer'] else 'no'}\n"

random.seed(42)
def build_prompt(train_data):
    yes_examples = [ex for ex in train_data if ex['answer'] == True]
    no_examples = [ex for ex in train_data if ex['answer'] == False]

    yes_examples = random.sample(yes_examples, 4)
    no_examples = random.sample(no_examples, 4)

    prompt = ""
    for y, n in zip(yes_examples, no_examples):
        prompt += format_example(y) + "\n"
        prompt += format_example(n) + "\n"

    return prompt

def query_model(prompt, question, passage):
    full_prompt = prompt + f"""
    Answer the question using ONLY 'yes'or 'no'. Do not explain.
    Question: {question}
    Passage: {passage}  
    Answer:
    """

    response = client.chat.completions.create(
        model= "gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.0
    )
    output = response.choices[0].message.content.strip().lower()
    output = output.strip()

    if output.startswith("yes"):
        return "yes"
    elif output.startswith("no"):
        return "no"
    else:
        return "unknown"
    

def evaluation():
    prompt = build_prompt(train_data)

    correct = 0
    total = 30

    for i in range(total):
        example = val_data[i]

        pred = query_model(prompt, example['question'], example['passage'])
        label = "yes" if example['answer'] else "no"

        if pred == label:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluation()
    