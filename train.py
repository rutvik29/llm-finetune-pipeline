"""Main fine-tuning entry point."""
import argparse
from src.config import TrainingConfig
from src.data.loader import load_and_format_dataset
from src.model.loader import load_model_and_tokenizer
from src.model.lora import apply_lora
from src.trainer.sft_trainer import run_sft_training


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Pipeline")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--quantization", type=str, default="4bit", choices=["none", "4bit", "8bit"])
    parser.add_argument("--output_dir", type=str, default="./outputs/model")
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    print(f"Loading model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)
    dataset = load_and_format_dataset(config, tokenizer)
    run_sft_training(model, tokenizer, dataset, config)
    print("Training complete!")
