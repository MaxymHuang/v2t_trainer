"""
Test Smart Home Voice Model
Lightweight inference optimized for M2000 deployment
"""

import torch
import librosa
import argparse
import time
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

from config import DeploymentConfig

class SmartHomeVoiceModel:
    """Lightweight voice model for smart home commands"""
    
    def __init__(self, model_path="./lightweight-whisper"):
        self.model_path = model_path
        self.config = DeploymentConfig()
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.load_model()
    
    def load_model(self):
        """Load model with memory optimization"""
        print(f"🤖 Loading model from {self.model_path}")
        print(f"📱 Device: {self.device}")
        
        if not Path(self.model_path).exists():
            print(f"❌ Model not found at {self.model_path}")
            print("💡 Train a model first: python train.py")
            return False
        
        try:
            # Load processor with proper language/task configuration
            self.processor = WhisperProcessor.from_pretrained(
                self.model_path,
                language="english",
                task="transcribe"
            )
            
            # Load model with optimizations (use FP32 to match training)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Match training configuration (FP32)
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Configure model for inference
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
            self.model.config.use_cache = True
            
            # Optimize for inference
            self.model.eval()
            
            if self.config.compile_model and hasattr(torch, 'compile'):
                print("⚡ Compiling model for faster inference...")
                self.model = torch.compile(self.model)
            
            # Memory check
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated(0) / 1e6
                print(f"💾 GPU Memory: {memory_mb:.0f} MB")
                
                if memory_mb > self.config.max_memory_gb * 1000:
                    print("⚠️  High memory usage - consider quantization")
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file to text"""
        if self.model is None:
            return None
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            if len(audio) == 0:
                return "ERROR: Empty audio file"
            
            # Preprocess
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Move to device and ensure proper data type (FP32)
            if self.device == "cuda":
                inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
            
            # Generate transcription
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=50,  # Smart home commands are short
                    do_sample=False,    # Deterministic for commands
                    num_beams=1,        # Fast greedy decoding
                    language="en",      # Force English
                    task="transcribe",  # Force transcription (not translation)
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            inference_time = time.time() - start_time
            
            return {
                "text": transcription.strip(),
                "inference_time": inference_time,
                "audio_duration": len(audio) / 16000
            }
            
        except Exception as e:
            return f"ERROR: {e}"
    
    def test_commands(self, test_dir="recordings"):
        """Test model on recorded commands"""
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"❌ Test directory not found: {test_dir}")
            return
        
        audio_files = list(test_path.glob("*.wav"))
        if len(audio_files) == 0:
            print(f"❌ No audio files found in {test_dir}")
            return
        
        print(f"🧪 Testing model on {len(audio_files)} files")
        print("=" * 60)
        
        correct = 0
        total_time = 0
        
        for audio_file in sorted(audio_files):
            # Find corresponding text file
            text_file = audio_file.with_suffix('.txt')
            expected = ""
            
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    expected = f.read().strip().lower()
            
            # Transcribe
            result = self.transcribe_audio(audio_file)
            
            if isinstance(result, dict):
                predicted = result["text"].lower()
                inference_time = result["inference_time"]
                total_time += inference_time
                
                # Check accuracy
                is_correct = predicted.strip() == expected.strip()
                if is_correct:
                    correct += 1
                
                print(f"📁 {audio_file.name}")
                print(f"   Expected: \"{expected}\"")
                print(f"   Got:      \"{predicted}\"")
                print(f"   ⏱️  Time: {inference_time:.3f}s")
                print(f"   {'✅' if is_correct else '❌'}")
                print()
            else:
                print(f"❌ Error with {audio_file.name}: {result}")
        
        # Summary
        accuracy = (correct / len(audio_files)) * 100
        avg_time = total_time / len(audio_files)
        
        print("📊 Test Results:")
        print(f"   Accuracy: {correct}/{len(audio_files)} ({accuracy:.1f}%)")
        print(f"   Avg inference time: {avg_time:.3f}s")
        print(f"   Real-time factor: {avg_time:.2f}x")
        
        # Performance assessment
        if accuracy >= 95:
            print("🟢 Excellent performance!")
        elif accuracy >= 85:
            print("🟡 Good performance")
        elif accuracy >= 70:
            print("🟠 Fair performance - consider more training")
        else:
            print("🔴 Poor performance - retrain with more data")

def main():
    parser = argparse.ArgumentParser(description="Test smart home voice model")
    parser.add_argument("--model_path", "-m", default="./lightweight-whisper",
                       help="Path to trained model")
    parser.add_argument("--audio_file", "-a", 
                       help="Single audio file to transcribe")
    parser.add_argument("--test_dir", "-t", default="recordings",
                       help="Directory with test audio files")
    parser.add_argument("--benchmark", "-b", action="store_true",
                       help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Load model
    model = SmartHomeVoiceModel(args.model_path)
    
    if model.model is None:
        return
    
    if args.audio_file:
        # Single file transcription
        print(f"🎵 Transcribing: {args.audio_file}")
        result = model.transcribe_audio(args.audio_file)
        
        if isinstance(result, dict):
            print(f"📝 Result: \"{result['text']}\"")
            print(f"⏱️  Time: {result['inference_time']:.3f}s")
        else:
            print(f"❌ Error: {result}")
    
    elif args.benchmark:
        # Performance benchmark
        print("🏃 Running performance benchmark...")
        
        # Create test audio
        import numpy as np
        import soundfile as sf
        
        # Generate 3-second test audio
        test_audio = np.random.randn(3 * 16000).astype(np.float32) * 0.1
        test_file = "benchmark_test.wav"
        sf.write(test_file, test_audio, 16000)
        
        # Warm up
        print("🔥 Warming up...")
        for _ in range(3):
            model.transcribe_audio(test_file)
        
        # Benchmark
        times = []
        for i in range(10):
            result = model.transcribe_audio(test_file)
            if isinstance(result, dict):
                times.append(result['inference_time'])
        
        # Results
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"📊 Benchmark Results (10 runs):")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Min: {min_time:.3f}s")
        print(f"   Max: {max_time:.3f}s")
        print(f"   Real-time factor: {avg_time:.2f}x")
        
        # Cleanup
        Path(test_file).unlink()
    
    else:
        # Test on directory
        model.test_commands(args.test_dir)

if __name__ == "__main__":
    main() 