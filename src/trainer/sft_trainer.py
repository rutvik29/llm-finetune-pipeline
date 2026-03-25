"""Supervised Fine-Tuning with TRL + MLflow tracking."""
import mlflow
from trl import SFTTrainer
from transformers import TrainingArguments
from src.config import TrainingConfig


def run_sft_training(model, tokenizer, dataset, config: TrainingConfig):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        dataloader_num_workers=config.dataloader_num_workers,
        report_to="mlflow",
    )

    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment(config.experiment_name)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
        packing=True,
    )

    with mlflow.start_run(run_name=f"{config.model_name.split('/')[-1]}_lora_r{config.lora_r}"):
        mlflow.log_params({
            "model": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "epochs": config.epochs,
            "lr": config.learning_rate,
            "quantization": config.quantization,
        })
        trainer.train()
        mlflow.log_artifact(config.output_dir)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return trainer
