"""Dataset loading and chat-template formatting."""
from datasets import load_dataset
from src.config import TrainingConfig


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_alpaca(example: dict) -> dict:
    text = ALPACA_TEMPLATE.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", "")
    )
    return {"text": text}


def load_and_format_dataset(config: TrainingConfig, tokenizer):
    dataset = load_dataset(config.dataset, split="train")
    
    if "alpaca" in config.dataset.lower():
        dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    
    dataset = dataset.filter(lambda x: len(x["text"]) < config.max_seq_length * 4)
    return dataset
