"""
Data Preparation for Smart Home Voice Commands
Converts audio recordings and transcripts into training format
"""

import os
import pandas as pd
import librosa
import soundfile as sf
import argparse
from pathlib import Path

def process_audio_files(input_dir, output_dir="dataset"):
    """Process raw audio files and create training dataset"""
    
    print("🎵 Smart Home Data Preparation")
    print("=" * 50)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(exist_ok=True)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Find audio and text files
    audio_files = []
    text_files = []
    
    # Common audio extensions
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    text_extensions = ['.txt']
    
    print(f"📁 Scanning {input_path}")
    
    for ext in audio_extensions:
        audio_files.extend(list(input_path.glob(f"*{ext}")))
        audio_files.extend(list(input_path.glob(f"**/*{ext}")))
    
    for ext in text_extensions:
        text_files.extend(list(input_path.glob(f"*{ext}")))
        text_files.extend(list(input_path.glob(f"**/*{ext}")))
    
    print(f"✅ Found {len(audio_files)} audio files")
    print(f"✅ Found {len(text_files)} text files")
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        print("💡 Make sure you have .wav, .mp3, or other audio files")
        return
    
    # Process files
    metadata = []
    processed_count = 0
    
    for audio_file in audio_files:
        # Find corresponding text file
        text_file = None
        base_name = audio_file.stem
        
        # Try different text file naming patterns
        for txt in text_files:
            if txt.stem == base_name or txt.stem in base_name or base_name in txt.stem:
                text_file = txt
                break
        
        if text_file is None:
            print(f"⚠️  No text file found for {audio_file.name}")
            continue
        
        # Read transcription
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
        except Exception as e:
            print(f"❌ Error reading {text_file}: {e}")
            continue
        
        if not transcription:
            print(f"⚠️  Empty transcription in {text_file.name}")
            continue
        
        # Process audio
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # Calculate duration
            duration = len(audio) / sr
            
            # Skip very short or very long clips
            if duration < 0.5 or duration > 10.0:
                print(f"⚠️  Skipping {audio_file.name} (duration: {duration:.1f}s)")
                continue
            
            # Generate output filename
            output_filename = f"{processed_count:06d}_{audio_file.stem}.wav"
            output_audio_path = audio_dir / output_filename
            
            # Save processed audio (16kHz, mono)
            sf.write(output_audio_path, audio, 16000)
            
            # Add to metadata
            metadata.append({
                'file': str(Path("audio") / output_filename),
                'transcription': transcription.lower().strip(),
                'duration': duration,
                'original_file': str(audio_file.name)
            })
            
            processed_count += 1
            print(f"✅ Processed: {audio_file.name} -> {output_filename} ({duration:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            continue
    
    if len(metadata) == 0:
        print("❌ No valid audio-text pairs found!")
        return
    
    # Create metadata.csv
    df = pd.DataFrame(metadata)
    metadata_path = output_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {len(df)} files")
    print(f"   📁 Output directory: {output_path}")
    print(f"   📄 Metadata: {metadata_path}")
    print(f"   🎵 Audio files: {audio_dir}")
    
    # Dataset statistics
    avg_duration = df['duration'].mean()
    total_duration = df['duration'].sum()
    print(f"\n📈 Dataset Statistics:")
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"   Duration range: {df['duration'].min():.1f}s - {df['duration'].max():.1f}s")
    
    # Sample commands
    print(f"\n🔍 Sample Commands:")
    for i, row in df.head(3).iterrows():
        print(f"   \"{row['transcription']}\" ({row['duration']:.1f}s)")
    
    print(f"\n✅ Data preparation complete!")
    print(f"💡 Next step: python train.py")

def create_sample_commands():
    """Create sample smart home commands for training"""
    
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
    
    print("📝 Sample Smart Home Commands")
    print("=" * 50)
    print("Record yourself saying these commands:")
    print()
    
    for i, command in enumerate(commands, 1):
        print(f"{i:2d}. \"{command}\"")
    
    print()
    print("💡 Tips:")
    print("   - Speak clearly and naturally")
    print("   - Record in a quiet environment")
    print("   - Each recording should be 2-6 seconds")
    print("   - Save as: command_001.wav, command_001.txt, etc.")
    print("   - Put the exact text in the .txt file")

def main():
    parser = argparse.ArgumentParser(description="Prepare smart home voice data")
    parser.add_argument("--input_dir", "-i", default="recordings", 
                       help="Directory with audio and text files")
    parser.add_argument("--output_dir", "-o", default="dataset",
                       help="Output directory for processed data")
    parser.add_argument("--sample_commands", action="store_true",
                       help="Show sample commands to record")
    
    args = parser.parse_args()
    
    if args.sample_commands:
        create_sample_commands()
    else:
        process_audio_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 