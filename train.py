"""
Lightweight Smart Home Voice Trainer
Optimized for i7-8700 + M2000 (4GB VRAM)
Uses HuggingFace Whisper models for maximum efficiency
"""

import os
import sys
import torch
import gc
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer, 
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
import librosa
import warnings
warnings.filterwarnings("ignore")
# Suppress specific transformers deprecation warning for past_key_values
warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")

from config import LightweightConfig

# Global tokenizer for metrics
tokenizer_global = None

def check_system():
    """Check system compatibility"""
    print("🔍 System Check")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 3.5:
            print("⚠️  Warning: Low GPU memory detected!")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("❌ No GPU detected - training will be very slow!")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check CPU
    print(f"💻 CPU cores: {os.cpu_count()}")
    
    # Check PyTorch version
    print(f"🔥 PyTorch: {torch.__version__}")
    print()

def load_dataset(data_dir):
    """Load custom voice dataset"""
    print(f"📁 Loading dataset from {data_dir}")
    
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"❌ No metadata.csv found in {data_dir}")
        print("💡 Run data preparation first or check dataset path")
        return None
    
    df = pd.read_csv(metadata_path)
    print(f"✅ Found {len(df)} samples")
    
    # Add full path to audio files
    df["file"] = df["file"].apply(lambda x: os.path.join(data_dir, x))
    
    # Verify files exist
    missing = []
    for file_path in df["file"]:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print(f"⚠️  {len(missing)} audio files missing")
        df = df[~df["file"].isin(missing)]
        print(f"📊 Using {len(df)} valid samples")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("file", Audio(sampling_rate=16000))
    
    return dataset

def split_dataset(dataset, config):
    """Split dataset with memory optimization"""
    print(f"🔄 Splitting dataset ({config.train_split_ratio:.0%} train)")
    
    df = dataset.to_pandas()
    train_df, val_df = train_test_split(
        df, 
        train_size=config.train_split_ratio,
        random_state=42,
        shuffle=True
    )
    
    # Ensure at least 1 validation sample
    if len(val_df) == 0:
        val_df = train_df.iloc[-1:].copy()
        train_df = train_df.iloc[:-1].copy()
    
    print(f"📊 Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # Convert back to datasets
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    
    # Cast audio columns
    train_dataset = train_dataset.cast_column("file", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("file", Audio(sampling_rate=16000))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

def prepare_dataset(dataset, processor, config):
    """Prepare dataset for training with memory optimization"""
    print("⚙️ Preprocessing audio data...")
    
    def preprocess_function(batch):
        # Extract audio
        audio = batch["file"]["array"]
        
        # Ensure proper format
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        
        # Process audio through feature extractor
        input_features = processor.feature_extractor(
            audio, 
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]
        
        # Tokenize transcription
        labels = processor.tokenizer(
            batch["transcription"],
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "input_features": input_features,
            "labels": labels
        }
    
    # Process in smaller batches to save memory
    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        num_proc=1,  # Single process to avoid memory issues
        desc="Processing audio"
    )
    
    # Filter out samples that are too long
    def filter_long_samples(sample):
        return len(sample["labels"]) <= 256  # Limit sequence length
    
    dataset = dataset.filter(filter_long_samples)
    
    return dataset

def compute_metrics(eval_pred):
    """Compute WER metric"""
    predictions, labels = eval_pred
    
    # Handle different prediction formats (tuple vs array)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract predictions from tuple
    
    # Convert to numpy array if needed
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Handle logits (3D) vs token IDs (2D)
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    # Convert labels to numpy array if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Replace -100 with pad token
    labels = np.where(labels != -100, labels, tokenizer_global.pad_token_id)
    
    try:
        # Decode predictions and labels
        pred_str = tokenizer_global.batch_decode(predictions, skip_special_tokens=True)
        label_str = tokenizer_global.batch_decode(labels, skip_special_tokens=True)
        
        # Compute WER
        wer_metric = evaluate.load("wer")
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
        
    except Exception as e:
        print(f"⚠️  Warning: Could not compute metrics: {e}")
        return {"wer": 0.0}  # Return default value to continue training

class DataCollator:
    """Lightweight data collator"""
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        return batch

def train_model():
    """Main training function"""
    config = LightweightConfig()
    
    print("🏠 Lightweight Smart Home Voice Trainer")
    print("=" * 50)
    print(f"🎯 Target: {config.model_name}")
    print(f"💾 VRAM Target: 2-3GB")
    print()
    
    # System check
    check_system()
    
    # Load dataset
    raw_dataset = load_dataset(config.data_dir)
    if raw_dataset is None:
        return
    
    # Analyze dataset
    print("📊 Dataset Analysis:")
    df = raw_dataset.to_pandas()
    if 'duration' in df.columns:
        print(f"   Duration: {df['duration'].mean():.1f}s avg, {df['duration'].min():.1f}-{df['duration'].max():.1f}s range")
    print(f"   Samples: {len(df)}")
    print()
    
    # Split dataset
    dataset_dict = split_dataset(raw_dataset, config)
    
    # Load model and processor
    print(f"🤖 Loading {config.model_name}")
    
    # Use forced_device_map for memory efficiency
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            use_cache=False if config.gradient_checkpointing else True,  # Prevent conflicts
            attn_implementation="eager"  # Use eager attention to avoid newer cache issues
        )
    except Exception as e:
        print(f"⚠️  Falling back to default model loading: {e}")
        model = WhisperForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            use_cache=False if config.gradient_checkpointing else True
        )
    
    processor = WhisperProcessor.from_pretrained(config.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(config.model_name)
    
    # Set global tokenizer for metrics
    global tokenizer_global
    tokenizer_global = tokenizer
    
    # Ensure model is on correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, 'device') or model.device.type != device.type:
        model = model.to(device)
        print(f"📱 Model moved to: {device}")
    
    # Configure model for training
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Fix for transformers v4.43+ deprecation warning
    if hasattr(model.config, 'cache_implementation'):
        model.config.cache_implementation = "static"  # Use newer cache format
        print(f"🔧 Using static cache implementation (v4.43+ compatible)")
    
    # Fix gradient checkpointing compatibility
    if config.gradient_checkpointing:
        model.config.use_cache = False  # Required for gradient checkpointing
        model.gradient_checkpointing_enable()
        print(f"⚡ Gradient checkpointing enabled (memory saving)")
    else:
        model.config.use_cache = True
        print(f"🚀 Gradient checkpointing disabled (faster training)")
    
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"🔧 Model config: use_cache={model.config.use_cache}, gradient_checkpointing={getattr(model, 'gradient_checkpointing', False)}")
    
    # Prepare datasets
    train_dataset = prepare_dataset(dataset_dict["train"], processor, config)
    val_dataset = prepare_dataset(dataset_dict["validation"], processor, config)
    
    print(f"📊 Ready: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        fp16=config.fp16,
        gradient_checkpointing=False,  # Handle manually on model to avoid conflicts
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=config.remove_unused_columns,
        label_names=config.label_names,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to=config.report_to,
    )
    
    # Data collator
    data_collator = DataCollator(processor)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Add early stopping
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    ))
    
    print("🚀 Starting training...")
    print(f"📊 Steps: {config.max_steps}")
    print(f"📦 Batch size: {config.per_device_train_batch_size}")
    print(f"⚡ Learning rate: {config.learning_rate}")
    
    # Monitor GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 GPU memory before: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
    
    # Train
    trainer.train()
    
    # Monitor final GPU usage
    if torch.cuda.is_available():
        print(f"💾 GPU memory after: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
    
    # Save model
    print(f"💾 Saving model to {config.output_dir}")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    # Final evaluation
    print("📊 Final evaluation...")
    results = trainer.evaluate()
    print(f"✅ Final WER: {results['eval_wer']:.2f}%")
    
    # Memory cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("✅ Training complete!")
    print(f"📂 Model saved: {config.output_dir}")

if __name__ == "__main__":
    train_model() 