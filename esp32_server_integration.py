"""
ESP32 Server Integration with Custom Whisper Model
Modified version of your ESP32 server to use custom trained model
"""

from flask import Flask, request, jsonify
import os
import tempfile
import base64
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Import custom model inference
try:
    from custom_whisper_inference import transcribe_with_custom_model
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  Custom model not available, falling back to OpenAI Whisper")
    CUSTOM_MODEL_AVAILABLE = False

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio_file(audio_file_path):
    """Process audio file and return transcription"""
    try:
        if CUSTOM_MODEL_AVAILABLE:
            # Use custom trained model
            print("🤖 Using custom trained model")
            result = transcribe_with_custom_model(audio_file_path)
            
            if result.startswith("ERROR:"):
                print(f"❌ Custom model error: {result}")
                return {"error": f"Custom model error: {result}"}
            
            return {
                "transcription": result,
                "model": "custom_whisper",
                "confidence": "high"  # Custom model trained for your voice
            }
        else:
            # Fallback to OpenAI Whisper (your original implementation)
            print("🔄 Using OpenAI Whisper (fallback)")
            import whisper
            
            model = whisper.load_model("base")
            result = model.transcribe(audio_file_path)
            
            return {
                "transcription": result["text"],
                "model": "openai_whisper",
                "confidence": "medium"
            }
            
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return {"error": f"Processing error: {str(e)}"}

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handle audio transcription requests"""
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Process the audio file
            result = process_audio_file(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if "error" in result:
                return jsonify(result), 500
            else:
                return jsonify(result)
                
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "custom" if CUSTOM_MODEL_AVAILABLE else "openai_fallback"
    
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "endpoints": ["/transcribe", "/health"]
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        "message": "ESP32 Voice-to-Text Server",
        "model": "custom_whisper" if CUSTOM_MODEL_AVAILABLE else "openai_whisper",
        "usage": {
            "endpoint": "/transcribe",
            "method": "POST",
            "content_type": "multipart/form-data",
            "field": "audio",
            "supported_formats": list(ALLOWED_EXTENSIONS)
        },
        "features": [
            "Custom trained model for smart home commands",
            "Optimized for your voice",
            "Fast inference on M2000 GPU",
            "Fallback to OpenAI Whisper if needed"
        ]
    })

if __name__ == '__main__':
    print("🚀 ESP32 Voice-to-Text Server")
    print("=" * 50)
    print(f"🤖 Model: {'Custom Whisper' if CUSTOM_MODEL_AVAILABLE else 'OpenAI Whisper (fallback)'}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📏 Max file size: {MAX_CONTENT_LENGTH / (1024*1024):.0f}MB")
    print(f"🎵 Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print()
    print("🌐 Server starting on http://localhost:5000")
    print("📋 Endpoints:")
    print("   GET  / - Home page with usage info")
    print("   POST /transcribe - Transcribe audio file")
    print("   GET  /health - Health check")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False) 