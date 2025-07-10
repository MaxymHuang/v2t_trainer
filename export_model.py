"""
Export Custom Model for ESP32 Deployment
Creates a deployment-ready version of your trained model
"""

import torch
import shutil
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import warnings
warnings.filterwarnings("ignore")

def export_model_for_deployment():
    """Export trained model for ESP32 deployment"""
    
    print("Exporting Model for ESP32 Deployment")
    print("=" * 50)
    
    # Paths
    trained_model_path = Path("./lightweight-whisper")
    deployment_path = Path("./deployed-model")
    
    if not trained_model_path.exists():
        print(f"❌ Trained model not found at {trained_model_path}")
        print("💡 Train a model first: python train.py")
        return False
    
    # Create deployment directory
    deployment_path.mkdir(exist_ok=True)
    
    print(f"📂 Loading model from: {trained_model_path}")
    
    try:
        # Load the trained model
        model = WhisperForConditionalGeneration.from_pretrained(
            trained_model_path,
            torch_dtype=torch.float32,  # Use FP32 for deployment
            use_cache=True
        )
        
        processor = WhisperProcessor.from_pretrained(trained_model_path)
        
        # Configure for deployment
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = True
        
        # Save deployment model
        print(f"💾 Saving to: {deployment_path}")
        model.save_pretrained(deployment_path)
        processor.save_pretrained(deployment_path)
        
        # Create model info file
        model_info = {
            "model_type": "whisper-base",
            "language": "english",
            "task": "transcribe",
            "parameters": sum(p.numel() for p in model.parameters()),
            "trained_for": "smart_home_commands"
        }
        
        import json
        with open(deployment_path / "model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2)
        
        print("Model exported successfully!")
        print(f"Deployment model: {deployment_path}")
        
        # Copy to ESP32 project if it exists
        esp32_project = Path("../v2t")
        if esp32_project.exists():
            esp32_model_path = esp32_project / "custom_model"
            esp32_model_path.mkdir(exist_ok=True)
            
            print(f"Copying to ESP32 project: {esp32_model_path}")
            shutil.copytree(deployment_path, esp32_model_path, dirs_exist_ok=True)
            print("Model copied to ESP32 project!")
        
        return True
        
    except Exception as e:
        print(f"Error exporting model: {e}")
        return False

def create_inference_script():
    """Create inference script for ESP32 server"""
    
    inference_script = '''"""
Custom Whisper Model Inference for ESP32 Server
Replace OpenAI Whisper with your custom trained model
"""

import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import warnings
warnings.filterwarnings("ignore")

class CustomWhisperModel:
    def __init__(self, model_path="./custom_model"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.load_model()
    
    def load_model(self):
        """Load custom trained model"""
        try:
            print(f"Loading custom model from {self.model_path}")
            
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(
                self.model_path,
                language="english",
                task="transcribe"
            )
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Configure for inference
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
            self.model.config.use_cache = True
            self.model.eval()
            
            print(f"Custom model loaded successfully")
            print(f"Device: {self.device}")
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated(0) / 1e6
                print(f"GPU Memory: {memory_mb:.0f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return False
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio data to text"""
        if self.model is None:
            return "ERROR: Model not loaded"
        
        try:
            # Ensure audio is mono and correct sample rate
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Preprocess
            inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt")
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    language="en",
                    task="transcribe",
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            return f"ERROR: {e}"

# Global model instance
custom_model = None

def get_custom_model():
    """Get or create custom model instance"""
    global custom_model
    if custom_model is None:
        custom_model = CustomWhisperModel()
    return custom_model

def transcribe_with_custom_model(audio_file_path):
    """Transcribe audio file using custom model"""
    model = get_custom_model()
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        
        # Transcribe
        result = model.transcribe_audio(audio, sr)
        
        return result
        
    except Exception as e:
        return f"ERROR: {e}"
'''
    
    # Save inference script
    script_path = Path("custom_whisper_inference.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(inference_script)
    
    print(f"Inference script created: {script_path}")
    return script_path

def main():
    print("ESP32 Custom Model Deployment")
    print("=" * 50)
    
    # Export model
    if export_model_for_deployment():
        # Create inference script
        create_inference_script()
        
        print("\nDeployment ready!")
        print("\nNext steps:")
        print("1. Copy 'deployed-model' to your ESP32 project")
        print("2. Copy 'custom_whisper_inference.py' to your ESP32 project")
        print("3. Update your server.py to use the custom model")
        print("4. Test with your ESP32 device")
        
        print("\nIntegration example:")
        print("from custom_whisper_inference import transcribe_with_custom_model")
        print("result = transcribe_with_custom_model('audio.wav')")
    else:
        print("Export failed")

if __name__ == "__main__":
    main()
