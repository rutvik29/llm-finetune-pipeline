"""LoRA adapter injection using PEFT."""
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from src.config import TrainingConfig


def apply_lora(model, config: TrainingConfig):
    if config.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
