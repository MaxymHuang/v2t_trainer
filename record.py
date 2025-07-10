"""
Simple Voice Recorder for Smart Home Commands
Optimized for quick recording sessions
"""

import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

def record_commands():
    """Interactive voice recording for smart home commands"""
    
    print("🎤 Smart Home Voice Recorder")
    print("=" * 50)
    
    # Setup
    sample_rate = 16000  # Whisper standard
    channels = 1         # Mono
    duration = 6         # Max recording length
    
    # Create recordings directory
    recordings_dir = Path("recordings")
    recordings_dir.mkdir(exist_ok=True)
    
    print(f"📁 Saving to: {recordings_dir}")
    print(f"🎵 Sample rate: {sample_rate}Hz")
    print(f"⏱️  Max duration: {duration}s")
    print()
    
    # Sample commands
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
    
    # Check audio devices
    print("🔊 Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"   {i}: {device['name']}")
    print()
    
    # Test audio
    print("🧪 Testing audio...")
    try:
        test_duration = 1.0
        test_audio = sd.rec(int(test_duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=channels, 
                           dtype='float32')
        sd.wait()
        
        # Check audio level
        rms = np.sqrt(np.mean(test_audio**2))
        if rms < 0.01:
            print("⚠️  Audio level very low - check microphone!")
        elif rms > 0.5:
            print("⚠️  Audio level very high - may clip!")
        else:
            print("✅ Audio level looks good")
    except Exception as e:
        print(f"⚠️  Audio test failed: {e}")
    
    print()
    
    # Recording loop
    command_index = 0
    
    while True:
        # Show current command
        if command_index < len(commands):
            current_command = commands[command_index]
            print(f"📝 Command {command_index + 1}/{len(commands)}: \"{current_command}\"")
            print("   Press Enter to record, 's' to skip, 'c' for custom, 'q' to quit")
        else:
            print("📝 All commands completed! Add custom commands or quit.")
            print("   Press Enter for custom command, 'q' to quit")
        
        # Get user input
        user_input = input("   > ").strip().lower()
        
        if user_input == 'q':
            break
        elif user_input == 's' and command_index < len(commands):
            command_index += 1
            continue
        elif user_input == 'c' or (user_input == '' and command_index >= len(commands)):
            custom_command = input("   Enter custom command: ").strip()
            if custom_command.lower() == 'quit':
                break
            elif custom_command:
                current_command = custom_command
            else:
                continue
        elif user_input == '' and command_index < len(commands):
            # Proceed with current command
            pass
        else:
            continue
        
        # Record audio
        print(f"🎤 Recording in 3... 2... 1...")
        time.sleep(1)
        print("🔴 RECORDING... (speak now)")
        
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=channels, 
                      dtype='float32')
        sd.wait()
        
        print("⏹️  Recording stopped")
        
        # Check if audio was captured
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            print("❌ No audio detected - try again")
            continue
        
        # Post-recording options (no auto-playback)
        while True:
            print("   Enter to save, 'r' to retry, 'p' to playback, 's' to skip")
            action = input("   > ").strip().lower()
            
            if action == '' or action == 'save':
                # Save the recording
                break
            elif action == 'r' or action == 'retry':
                # Retry recording
                break
            elif action == 'p' or action == 'playback':
                # Play back the recording
                print("🔊 Playing back...")
                sd.play(audio, sample_rate)
                sd.wait()
                continue
            elif action == 's' or action == 'skip':
                # Skip this recording
                if command_index < len(commands):
                    command_index += 1
                break
            else:
                print("❓ Invalid option. Try again.")
                continue
        
        # Handle the action
        if action == 'r' or action == 'retry':
            continue  # Go back to recording
        elif action == 's' or action == 'skip':
            continue  # Move to next command
        elif action != '' and action not in ['save']:
            continue  # Invalid action, retry
        
        # If we get here, user wants to save
        
        # Save files
        timestamp = int(time.time())
        audio_filename = f"command_{timestamp:010d}.wav"
        text_filename = f"command_{timestamp:010d}.txt"
        
        audio_path = recordings_dir / audio_filename
        text_path = recordings_dir / text_filename
        
        # Trim silence and save audio
        # Simple silence trimming
        threshold = 0.01
        start_idx = 0
        end_idx = len(audio)
        
        # Find start
        for i in range(len(audio)):
            if abs(audio[i]) > threshold:
                start_idx = max(0, i - 1600)  # 0.1s buffer
                break
        
        # Find end
        for i in range(len(audio) - 1, -1, -1):
            if abs(audio[i]) > threshold:
                end_idx = min(len(audio), i + 1600)  # 0.1s buffer
                break
        
        trimmed_audio = audio[start_idx:end_idx]
        
        # Save files
        sf.write(audio_path, trimmed_audio, sample_rate)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(current_command.lower().strip())
        
        duration_recorded = len(trimmed_audio) / sample_rate
        print(f"✅ Saved: {audio_filename} ({duration_recorded:.1f}s)")
        
        # Move to next command (only if we're going through the preset list)
        if command_index < len(commands):
            command_index += 1
        
        print()
    
    # Summary
    audio_files = list(recordings_dir.glob("*.wav"))
    text_files = list(recordings_dir.glob("*.txt"))
    
    print(f"\n📊 Recording Session Complete!")
    print(f"   🎵 Audio files: {len(audio_files)}")
    print(f"   📄 Text files: {len(text_files)}")
    print(f"   📁 Location: {recordings_dir}")
    
    if len(audio_files) > 0:
        print(f"\n💡 Next steps:")
        print(f"   1. python prepare_data.py -i recordings")
        print(f"   2. python train.py")

def main():
    try:
        record_commands()
    except KeyboardInterrupt:
        print("\n\n⏹️  Recording stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 