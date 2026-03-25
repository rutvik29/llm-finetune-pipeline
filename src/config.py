"""Training configuration dataclass."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantization: str = "4bit"  # none | 4bit | 8bit
    
    # Dataset
    dataset: str = "tatsu-lab/alpaca"
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    
    # Output
    output_dir: str = "./outputs/model"
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "llm-finetune"
    
    # Hardware
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
