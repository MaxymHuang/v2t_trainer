# 🏠 Lightweight Smart Home Voice Trainer

**M2000 Optimized** - Train your own voice recognition for smart home commands using HuggingFace Whisper models.

## 🎯 System Requirements

- **CPU**: i7-8700 or similar (6+ cores recommended)
- **GPU**: M2000 (4GB VRAM) or any GPU with 3+ GB VRAM
- **RAM**: 8GB+ system RAM
- **Storage**: 2GB for models + your recordings

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Check system compatibility
python train.py --help
```

### 2. Record Your Voice Commands

```bash
# Show sample commands to record
python prepare_data.py --sample_commands

# Start recording session
python record.py
```

**Recording Tips:**
- Speak clearly and naturally
- Record in a quiet environment  
- Each command: 2-6 seconds
- Record 20+ commands for good results
- Record multiple variations of each command

### 3. Prepare Training Data

```bash
# Process your recordings
python prepare_data.py -i recordings -o dataset
```

### 4. Train Your Model

```bash
# Start training (optimized for M2000)
python train.py
```

**Training will:**
- Use `whisper-tiny` model (~500MB VRAM)
- Train for 1000 steps (15-30 minutes)
- Auto-save best checkpoint
- Monitor GPU memory usage

### 5. Test Your Model

```bash
# Test on your recordings
python test_model.py -t recordings

# Test single file
python test_model.py -a my_command.wav

# Performance benchmark
python test_model.py -b
```

## 📊 Expected Performance

### VRAM Usage (M2000 4GB)
- **Training**: 2-3GB VRAM
- **Inference**: 500MB - 1GB VRAM  
- **Buffer**: 1GB+ remaining

### Accuracy Targets
- **Excellent**: 95%+ (your voice, clean audio)
- **Good**: 85%+ (some noise/variations)
- **Fair**: 70%+ (need more training data)

### Speed
- **Training**: 15-30 minutes for 1000 steps
- **Inference**: <100ms per command (real-time)

## 🛠️ Advanced Configuration

### Model Size Options

Edit `config.py`:

```python
# Ultra-lightweight (39M params, ~500MB VRAM)
model_name = "openai/whisper-tiny"

# Lightweight (74M params, ~1GB VRAM)  
model_name = "openai/whisper-base"

# Higher quality (244M params, ~2.5GB VRAM)
model_name = "openai/whisper-small"
```

### Training Optimization

For **faster training** on M2000:
```python
per_device_train_batch_size = 8  # If you have extra VRAM
gradient_accumulation_steps = 2  # Reduce if batch size increased
```

For **lower memory usage**:
```python
per_device_train_batch_size = 2  # Reduce batch size
gradient_accumulation_steps = 8  # Increase to maintain effective batch
```

## 📁 Project Structure

```
v2t_trainer/
├── config.py              # Model configuration
├── train.py               # Main training script  
├── record.py              # Voice recording tool
├── prepare_data.py        # Data preprocessing
├── test_model.py          # Model testing/inference
├── requirements.txt       # Dependencies
├── recordings/            # Your voice recordings
├── dataset/               # Processed training data
└── lightweight-whisper/   # Trained model output
```

## 🎵 Recording Commands

### Suggested Smart Home Commands

**Lighting:**
- "turn on the living room lights"
- "turn off the kitchen lights"  
- "dim the lights to 50 percent"
- "turn on porch light"

**Climate:**
- "set temperature to 72 degrees"
- "turn on air conditioning"
- "turn off heating"

**Security:**
- "lock the front door"
- "unlock the back door"
- "arm security system"
- "disarm security system"

**Entertainment:**
- "turn on the tv"
- "play music in living room"
- "stop the music"
- "volume up"

**Appliances:**
- "start the dishwasher"
- "open garage door"
- "close garage door"

### Recording Tips

1. **Consistency**: Use the same microphone/environment
2. **Variations**: Record each command 2-3 times slightly differently
3. **Natural Speech**: Don't over-enunciate
4. **Background**: Some background noise is OK (makes model robust)
5. **Duration**: 2-6 seconds per command ideal

## 🔧 Troubleshooting

### Common Issues

**"No GPU detected"**
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch GPU: `python -c "import torch; print(torch.cuda.is_available())"`

**"Out of memory"**
- Reduce batch size in `config.py`
- Use `whisper-tiny` instead of `whisper-base`
- Close other GPU applications

**"Poor accuracy"**
- Record more training data (30+ commands)
- Check audio quality (no clipping/noise)
- Train longer (increase `max_steps`)

**"Audio device not found"**
- List devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Install audio drivers

### Performance Optimization

**For M2000 deployment:**
```python
# In config.py - DeploymentConfig
use_onnx = True          # 20-30% faster inference
quantize_model = True    # 50% less memory
max_memory_gb = 3.5      # Leave 0.5GB buffer
```

## 🚀 Deployment

### Export for Production

```bash
# Create optimized deployment model
python -c "
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained('./lightweight-whisper')
model.save_pretrained('./deployed-model', torch_dtype='float16')
"
```

### Integration Example

```python
from test_model import SmartHomeVoiceModel

# Load model
model = SmartHomeVoiceModel("./lightweight-whisper")

# Transcribe command
result = model.transcribe_audio("command.wav")
print(f"Command: {result['text']}")
```

## 📈 Model Comparison

| Model | Parameters | VRAM | Speed | Quality |
|-------|------------|------|-------|---------|
| tiny  | 39M        | 500MB| Fastest | Good |
| base  | 74M        | 1GB  | Fast    | Better |
| small | 244M       | 2.5GB| Medium  | Best |

**Recommendation for M2000**: Start with `whisper-tiny`, upgrade to `base` if needed.

## 🆘 Support

**Model not learning?**
- Check your recordings are clear
- Ensure text files match audio exactly
- Try recording more varied examples

**Want better accuracy?**
- Record 50+ commands instead of 20
- Include background noise variations
- Train for 2000+ steps

**Running out of VRAM?**
- Use `whisper-tiny` model
- Reduce batch size to 2
- Close other applications

## 🎉 Success Tips

1. **Quality over Quantity**: 20 clear recordings > 50 noisy ones
2. **Consistency**: Same person, same microphone  
3. **Patience**: Let it train completely (don't stop early)
4. **Testing**: Always test on new recordings, not training data
5. **Iterations**: Retrain with more data if accuracy is low

---

**🎯 Goal**: 95%+ accuracy on your voice for smart home commands in <1GB VRAM! 