"""
Model Optimization for M2000 Deployment
Creates optimized versions of trained models for production use
"""

import torch
import argparse
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import warnings
warnings.filterwarnings("ignore")

def optimize_model(model_path, output_path="./optimized-model"):
    """Create optimized model for M2000 deployment"""
    
    print("🔧 M2000 Model Optimization")
    print("=" * 50)
    
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Load model
    print(f"📂 Loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # Original model info
    original_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Original parameters: {original_params:,}")
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Optimization 1: FP16 conversion
    print("\n🔄 Converting to FP16...")
    model_fp16 = model.half()
    
    fp16_path = output_path / "fp16"
    fp16_path.mkdir(exist_ok=True)
    
    model_fp16.save_pretrained(fp16_path, torch_dtype=torch.float16)
    processor.save_pretrained(fp16_path)
    
    print(f"✅ FP16 model saved: {fp16_path}")
    
    # Optimization 2: Quantized INT8 (requires optimum)
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        
        print("\n🔄 Creating ONNX + INT8 quantized model...")
        
        # Convert to ONNX
        onnx_path = output_path / "onnx"
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            export=True,
            use_cache=False
        )
        onnx_model.save_pretrained(onnx_path)
        processor.save_pretrained(onnx_path)
        
        print(f"✅ ONNX model saved: {onnx_path}")
        
        # Quantize to INT8
        quantized_path = output_path / "quantized"
        quantizer = ORTQuantizer.from_pretrained(onnx_path)
        
        # Dynamic quantization (good balance of speed/accuracy)
        qconfig = AutoQuantizationConfig.dynamic()
        quantizer.quantize(save_dir=quantized_path, quantization_config=qconfig)
        
        print(f"✅ Quantized model saved: {quantized_path}")
        
    except ImportError:
        print("⚠️  Optimum not installed - skipping ONNX optimization")
        print("   Install with: pip install optimum[onnxruntime]")
    
    # Optimization 3: Pruned model (experimental)
    print("\n🔄 Creating pruned model...")
    
    # Simple magnitude-based pruning
    pruned_model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Prune small weights (10% threshold)
    pruning_threshold = 0.1
    pruned_params = 0
    total_params = 0
    
    with torch.no_grad():
        for name, param in pruned_model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                # Calculate threshold for this layer
                threshold = pruning_threshold * param.abs().max()
                
                # Create mask
                mask = param.abs() > threshold
                
                # Apply mask
                param.data = param.data * mask.float()
                
                # Stats
                pruned_count = (~mask).sum().item()
                total_count = mask.numel()
                
                pruned_params += pruned_count
                total_params += total_count
    
    pruning_ratio = pruned_params / total_params * 100
    print(f"🗜️  Pruned {pruning_ratio:.1f}% of weights")
    
    # Save pruned model
    pruned_path = output_path / "pruned"
    pruned_path.mkdir(exist_ok=True)
    
    pruned_model.save_pretrained(pruned_path)
    processor.save_pretrained(pruned_path)
    
    print(f"✅ Pruned model saved: {pruned_path}")
    
    # Model size comparison
    print(f"\n📏 Model Size Comparison:")
    
    original_size = sum(f.stat().st_size for f in model_path.rglob('*.safetensors'))
    
    for opt_dir in ['fp16', 'pruned']:
        if (output_path / opt_dir).exists():
            opt_size = sum(f.stat().st_size for f in (output_path / opt_dir).rglob('*.safetensors'))
            reduction = (1 - opt_size/original_size) * 100
            print(f"   {opt_dir}: {opt_size/1e6:.1f}MB ({reduction:.1f}% smaller)")
    
    # VRAM estimation
    print(f"\n💾 Estimated VRAM Usage:")
    print(f"   Original: ~{original_params * 4 / 1e6:.0f}MB (FP32)")
    print(f"   FP16: ~{original_params * 2 / 1e6:.0f}MB")
    print(f"   Quantized: ~{original_params * 1 / 1e6:.0f}MB (if available)")
    
    # Performance recommendations
    print(f"\n🎯 M2000 Deployment Recommendations:")
    
    estimated_vram_fp16 = original_params * 2 / 1e6
    if estimated_vram_fp16 < 1000:  # < 1GB
        print("✅ Use FP16 model - excellent for M2000")
    elif estimated_vram_fp16 < 2000:  # < 2GB  
        print("🟡 Use FP16 model - good for M2000")
    else:
        print("🔴 Consider quantized model - high VRAM usage")
    
    print(f"\n📚 Usage Examples:")
    print(f"FP16 model:")
    print(f"  model = WhisperForConditionalGeneration.from_pretrained('{fp16_path}', torch_dtype=torch.float16)")
    
    if (output_path / "onnx").exists():
        print(f"\nONNX model:")
        print(f"  from optimum.onnxruntime import ORTModelForSpeechSeq2Seq")
        print(f"  model = ORTModelForSpeechSeq2Seq.from_pretrained('{output_path / 'onnx'}')")
    
    return True

def benchmark_models(optimized_dir="./optimized-model"):
    """Benchmark different model variants"""
    
    print("🏃 Model Benchmark")
    print("=" * 50)
    
    import time
    import numpy as np
    import soundfile as sf
    
    # Create test audio
    test_audio = np.random.randn(3 * 16000).astype(np.float32) * 0.1
    test_file = "benchmark_test.wav"
    sf.write(test_file, test_audio, 16000)
    
    optimized_path = Path(optimized_dir)
    
    models_to_test = []
    
    # Check available models
    for model_type in ['fp16', 'pruned']:
        model_path = optimized_path / model_type
        if model_path.exists():
            models_to_test.append((model_type, model_path))
    
    if len(models_to_test) == 0:
        print("❌ No optimized models found")
        print("💡 Run optimization first")
        return
    
    results = {}
    
    for model_name, model_path in models_to_test:
        print(f"\n🧪 Testing {model_name} model...")
        
        try:
            # Load model
            if model_name == 'onnx':
                from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
                model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path)
            else:
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if model_name == 'fp16' else torch.float32
                )
            
            processor = WhisperProcessor.from_pretrained(model_path)
            
            # Move to GPU if available
            if torch.cuda.is_available() and hasattr(model, 'to'):
                model = model.to('cuda')
            
            # Warm up
            for _ in range(3):
                inputs = processor(test_audio, sampling_rate=16000, return_tensors="pt")
                if torch.cuda.is_available() and hasattr(model, 'to'):
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = model.generate(inputs["input_features"], max_new_tokens=10)
            
            # Benchmark
            times = []
            for _ in range(10):
                inputs = processor(test_audio, sampling_rate=16000, return_tensors="pt")
                if torch.cuda.is_available() and hasattr(model, 'to'):
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                start_time = time.time()
                with torch.no_grad():
                    _ = model.generate(inputs["input_features"], max_new_tokens=10)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            results[model_name] = avg_time
            
            print(f"   Average inference: {avg_time:.3f}s")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated(0) / 1e6
                print(f"   GPU memory: {memory_mb:.0f}MB")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n📊 Benchmark Summary:")
        best_model = min(results, key=results.get)
        
        for model_name, avg_time in sorted(results.items(), key=lambda x: x[1]):
            speedup = results[best_model] / avg_time if model_name != best_model else 1.0
            print(f"   {model_name}: {avg_time:.3f}s ({speedup:.1f}x)")
        
        print(f"\n🏆 Fastest: {best_model}")
    
    # Cleanup
    Path(test_file).unlink()

def main():
    parser = argparse.ArgumentParser(description="Optimize models for M2000 deployment")
    parser.add_argument("--model_path", "-m", default="./lightweight-whisper",
                       help="Path to trained model")
    parser.add_argument("--output_path", "-o", default="./optimized-model",
                       help="Output directory for optimized models")
    parser.add_argument("--benchmark", "-b", action="store_true",
                       help="Benchmark optimized models")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_models(args.output_path)
    else:
        optimize_model(args.model_path, args.output_path)

if __name__ == "__main__":
    main() 