"""
Lightweight Smart Home Voice Trainer Configuration
Optimized for i7-8700 + M2000 (4GB VRAM)
Target: 2-3GB VRAM usage when deployed
"""

class LightweightConfig:
    # Model configuration - optimized for low VRAM
    model_name = "openai/whisper-base"  # ~74M parameters, ~1GB VRAM
    # Alternative: "openai/whisper-tiny" # ~39M parameters, ~500MB VRAM
    language = "english"
    task = "transcribe"
    
    # Dataset configuration
    data_dir = "dataset"
    train_split_ratio = 0.8  # 80% train, 20% validation
    max_input_length = 8.0  # Max audio length in seconds
    
    # Training configuration - M2000 optimized
    output_dir = "./lightweight-whisper"
    per_device_train_batch_size = 2   # Smaller batch for whisper-base + FP32
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 8   # Effective batch size = 16
    learning_rate = 1e-5              # Slightly higher for faster convergence
    warmup_steps = 100
    max_steps = 1000                  # Fewer steps for lightweight training
    
    # Memory optimization
    fp16 = False                      # Disabled due to gradient scaling conflicts (FP32 is more stable)
    gradient_checkpointing = False    # Disabled for stability (uses more VRAM but more reliable)
    dataloader_num_workers = 0        # Avoid multiprocessing overhead
    
    # Evaluation and saving
    evaluation_strategy = "steps"
    eval_steps = 100
    save_steps = 100
    logging_steps = 10
    load_best_model_at_end = True
    metric_for_best_model = "wer"
    greater_is_better = False
    save_total_limit = 2              # Keep only 2 checkpoints to save disk space
    
    # Advanced optimizations
    remove_unused_columns = False
    label_names = ["labels"]
    push_to_hub = False
    report_to = ["tensorboard"]
    
    # Early stopping
    early_stopping_patience = 3
    early_stopping_threshold = 0.01

class DeploymentConfig:
    """Configuration for deployment on M2000"""
    
    # Model optimization
    use_onnx = True                   # Convert to ONNX for faster inference
    quantize_model = True             # 8-bit quantization
    compile_model = True              # Torch compile for speed
    
    # Memory limits
    max_memory_gb = 3.5               # Leave 0.5GB buffer on M2000
    batch_size = 1                    # Single inference for real-time use
    
    # Performance
    num_threads = 4                   # Optimize for i7-8700 (6 cores)
    device_map = "auto"               # Let transformers handle device placement 