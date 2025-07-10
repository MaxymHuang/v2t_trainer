"""
Quick Setup for Lightweight Smart Home Voice Trainer
One-command setup for M2000 users
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking dependencies...")
    
    missing = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  No CUDA GPU detected")
            
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError:
        missing.append("datasets")
    
    try:
        import sounddevice
        print(f"✅ Audio recording ready")
    except ImportError:
        missing.append("sounddevice")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies ready!")
    return True

def show_commands():
    """Show sample commands for reference"""
    commands = [
        "turn on the living room lights",
        "turn off the kitchen lights", 
        "set bedroom temperature to 72 degrees",
        "lock the front door",
        "unlock the back door",
        "play music in the living room",
        "stop the music",
        "turn on the tv",
        "turn off the tv",
        "dim the lights to 50 percent",
        "set thermostat to 68 degrees",
        "open the garage door",
        "close the garage door",
        "turn on air conditioning",
        "turn off air conditioning",
        "arm security system",
        "disarm security system",
        "turn on porch light",
        "turn off porch light",
        "start the dishwasher",
    ]
    
    print("\n📝 Sample Smart Home Commands to Record:")
    print("=" * 50)
    
    for i, command in enumerate(commands, 1):
        print(f"{i:2d}. \"{command}\"")
    
    print("\n💡 Recording Tips:")
    print("   • Speak naturally (don't over-enunciate)")
    print("   • 2-6 seconds per command")
    print("   • Record 20+ commands for good results")
    print("   • Record multiple variations of each command")
    print("   • Use consistent microphone and environment")

def interactive_setup():
    """Interactive setup guide"""
    print("🏠 Lightweight Smart Home Voice Trainer Setup")
    print("=" * 60)
    print("Optimized for i7-8700 + M2000 (4GB VRAM)")
    print()
    
    # Check system
    if not check_requirements():
        return
    
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    
    print("\n🚀 Quick Start Guide:")
    print("1️⃣  Record your voice commands:")
    print("   python record.py")
    
    print("\n2️⃣  Prepare training data:")
    print("   python prepare_data.py -i recordings")
    
    print("\n3️⃣  Train your model:")
    print("   python train.py")
    
    print("\n4️⃣  Test your model:")
    print("   python test_model.py -t recordings")
    
    print("\n🔧 Optional - Optimize for deployment:")
    print("   python optimize_for_deployment.py")
    
    # Ask if they want to see sample commands
    response = input("\n📋 Show sample commands to record? (y/n): ").strip().lower()
    if response == 'y':
        show_commands()
    
    # Ask if they want to start recording now
    response = input("\n🎤 Start recording session now? (y/n): ").strip().lower()
    if response == 'y':
        print("\n🎬 Starting recorder...")
        try:
            subprocess.run([sys.executable, "record.py"], check=True)
        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped")
        except Exception as e:
            print(f"\n❌ Error starting recorder: {e}")
    
    print("\n📚 For detailed instructions, see: README.md")
    print("🆘 For troubleshooting, see the README.md troubleshooting section")

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick setup for smart home voice trainer")
    parser.add_argument("--install", action="store_true", 
                       help="Install dependencies first")
    parser.add_argument("--check", action="store_true",
                       help="Only check system requirements")
    parser.add_argument("--commands", action="store_true",
                       help="Show sample commands")
    
    args = parser.parse_args()
    
    if args.install:
        if install_dependencies():
            print("\n✅ Ready to proceed with setup!")
        else:
            return
    
    if args.check:
        check_requirements()
        return
    
    if args.commands:
        show_commands()
        return
    
    # Full interactive setup
    interactive_setup()

if __name__ == "__main__":
    main() 