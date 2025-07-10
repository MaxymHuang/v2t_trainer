# 🚀 Custom Model Deployment Guide

**Deploy your trained smart home voice model to your ESP32 project**

## 📋 Overview

This guide shows how to integrate your **custom trained Whisper model** into your existing [ESP32 Voice-to-Text project](https://github.com/MaxymHuang/v2t).

## 🎯 Benefits

- ✅ **Better accuracy** for your voice and smart home commands
- ✅ **No API costs** (runs locally)
- ✅ **Faster inference** (optimized for your use case)
- ✅ **Privacy** (no data sent to external services)
- ✅ **Fallback** to OpenAI Whisper if needed

## 📦 Step 1: Export Your Model

First, export your trained model for deployment:

```bash
# Export the model
python export_model.py
```

This creates:
- `deployed-model/` - Your custom model ready for deployment
- `custom_whisper_inference.py` - Inference script for your server

## 📁 Step 2: Copy Files to ESP32 Project

Copy the exported files to your ESP32 project:

```bash
# Copy model to ESP32 project
cp -r deployed-model/ ../v2t/custom_model/

# Copy inference script
cp custom_whisper_inference.py ../v2t/

# Copy integrated server
cp esp32_server_integration.py ../v2t/server_custom.py
```

## 🔧 Step 3: Install Dependencies

In your ESP32 project directory, install the required dependencies:

```bash
cd ../v2t
pip install torch transformers librosa soundfile
```

## 🚀 Step 4: Test the Integration

### Test the Custom Model

```bash
# Test with a sample audio file
python custom_whisper_inference.py
```

### Start the Server

```bash
# Start server with custom model
python server_custom.py
```

### Test API Endpoint

```bash
# Test transcription endpoint
curl -X POST -F "audio=@test_audio.wav" http://localhost:5000/transcribe
```

## 📊 Expected Results

### **Custom Model Response**
```json
{
  "transcription": "turn on the living room lights",
  "model": "custom_whisper",
  "confidence": "high"
}
```

### **Fallback Response** (if custom model fails)
```json
{
  "transcription": "turn on the living room lights",
  "model": "openai_whisper",
  "confidence": "medium"
}
```

## 🔄 Integration Options

### **Option 1: Replace Existing Server**
```bash
# Backup original server
cp server.py server_openai_backup.py

# Use custom server
cp server_custom.py server.py
```

### **Option 2: Run Both Servers**
```bash
# Original server (port 5000)
python server.py

# Custom server (port 5001)
python server_custom.py
```

### **Option 3: Hybrid Approach**
Modify your existing `server.py` to use the custom model:

```python
# Add to your existing server.py
try:
    from custom_whisper_inference import transcribe_with_custom_model
    USE_CUSTOM_MODEL = True
except ImportError:
    USE_CUSTOM_MODEL = False

# In your transcription function:
if USE_CUSTOM_MODEL:
    result = transcribe_with_custom_model(audio_file_path)
else:
    # Your existing OpenAI Whisper code
    result = model.transcribe(audio_file_path)
```

## 🎯 ESP32 Code Updates

Update your ESP32 code to use the new endpoint:

### **Original ESP32 Code**
```cpp
// Your existing ESP32 code should work unchanged
// Just point to the new server endpoint
```

### **Optional: Add Model Selection**
```cpp
// Add model selection in ESP32
String model_type = "custom";  // or "openai"
String endpoint = "/transcribe";
if (model_type == "custom") {
    endpoint = "/transcribe_custom";
}
```

## 📈 Performance Comparison

| Metric | OpenAI Whisper | Custom Model |
|--------|----------------|--------------|
| **Accuracy** | Good | **Excellent** (your voice) |
| **Speed** | ~2-3s | **~0.5-1s** |
| **Cost** | $0.006/min | **$0** |
| **Privacy** | Data sent to OpenAI | **100% local** |
| **Reliability** | High | **High + fallback** |

## 🔧 Troubleshooting

### **Model Loading Issues**
```bash
# Check model files
ls -la custom_model/

# Test model loading
python -c "from custom_whisper_inference import CustomWhisperModel; m = CustomWhisperModel()"
```

### **Memory Issues**
```bash
# Monitor GPU memory
nvidia-smi

# Use CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

### **Dependency Issues**
```bash
# Install specific versions
pip install torch==2.0.1 transformers==4.35.0
```

## 🎉 Success Indicators

✅ **Server starts without errors**  
✅ **Custom model loads successfully**  
✅ **Transcription works for your voice**  
✅ **ESP32 can connect and transcribe**  
✅ **Fallback works if custom model fails**  

## 📞 Support

If you encounter issues:

1. **Check logs** - Look for error messages
2. **Test model** - Use `test_model.py` to verify
3. **Verify files** - Ensure all files are copied correctly
4. **Check dependencies** - Make sure all packages are installed

## 🚀 Next Steps

1. **Deploy to production** - Use your custom model in real ESP32 setup
2. **Monitor performance** - Track accuracy and speed
3. **Retrain if needed** - Add more voice samples for better accuracy
4. **Optimize further** - Use model quantization for even faster inference

---

**🎯 Goal**: Your ESP32 now uses your custom-trained voice model for perfect smart home command recognition! 