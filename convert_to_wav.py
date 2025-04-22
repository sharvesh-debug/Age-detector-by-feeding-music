import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# First, we'll check for required libraries and install them if needed
try:
    import librosa
except ImportError:
    print("Installing librosa...")
    subprocess.run(["pip", "install", "librosa"], check=True)
    import librosa

# Check if ffmpeg is installed or set custom path
FFMPEG_PATH = "C:\\ffmpeg\\ffmpeg-2025-04-17-git-7684243fbe-full_build\\bin\\ffmpeg.exe"

def check_ffmpeg():
    # First try with the custom path
    try:
        result = subprocess.run([FFMPEG_PATH, "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True,
                               shell=True)
        if result.returncode == 0:
            print(f"✅ FFmpeg found at custom path: {FFMPEG_PATH}")
            return FFMPEG_PATH
    except Exception as e:
        print(f"❌ Custom FFmpeg path failed: {e}")
    
    # Then try with system path
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            print("✅ FFmpeg found in system PATH")
            return "ffmpeg"
    except Exception as e:
        print(f"❌ System FFmpeg check failed: {e}")
    
    return None

# Step 1: Create a function to check file format
def check_audio_file(filepath):
    """Check audio file type and if it can be read."""
    print(f"Diagnosing: {os.path.basename(filepath)}")
    
    # Try with librosa (works with various formats)
    try:
        y, sr = librosa.load(filepath, sr=None, duration=1)
        if len(y) > 0:
            print(f"✅ librosa.load successful: {len(y)} samples at {sr}Hz")
            return True
    except Exception as e:
        print(f"❌ librosa.load failed: {str(e)}")
    
    # Try with soundfile (for WAV files)
    try:
        import soundfile as sf
        data, samplerate = sf.read(filepath)
        if len(data) > 0:
            print(f"✅ soundfile.read successful: {len(data)} samples at {samplerate}Hz")
            return True
    except Exception as e:
        print(f"❌ soundfile.read failed: {str(e)}")
    
    print("❌ All methods failed to read this file")
    return False

# Step 2: Create a function to convert audio files to a standard format
def convert_to_standard_wav(input_file, output_dir, ffmpeg_path):
    """Convert an audio file to 16-bit PCM WAV format."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + "_converted.wav")
    
    # Get the absolute paths
    input_file_abs = os.path.abspath(input_file)
    output_file_abs = os.path.abspath(output_file)
    
    print(f"Converting: {input_file_abs} -> {output_file_abs}")
    
    # Check if input file exists
    if not os.path.exists(input_file_abs):
        print(f"❌ Input file not found: {input_file_abs}")
        return None
    
    # Use ffmpeg to convert the file
    try:
        cmd = [
            ffmpeg_path,
            "-y",                # Overwrite output file if it exists
            "-i", input_file_abs,  # Input file
            "-acodec", "pcm_s16le",  # 16-bit PCM encoding
            "-ar", "44100",      # 44.1 kHz sample rate 
            "-ac", "2",          # Stereo (change to 1 for mono if needed)
            output_file_abs      # Output file
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Converted: {filename}")
            return output_file_abs
        else:
            print(f"❌ FFmpeg conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return None

# Step 3: Function to extract features
def extract_features(y, sr):
    """Extract audio features from a loaded audio file."""
    features = {}
    
    # Extract features with error handling
    try:
        features["chroma_stft"] = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    except:
        features["chroma_stft"] = np.nan
        
    try:
        features["rmse"] = librosa.feature.rms(y=y).mean()
    except:
        features["rmse"] = np.nan
        
    try:
        features["spectral_centroid"] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    except:
        features["spectral_centroid"] = np.nan
        
    try:
        features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    except:
        features["spectral_bandwidth"] = np.nan
        
    try:
        features["rolloff"] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    except:
        features["rolloff"] = np.nan
        
    try:
        features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y).mean()
    except:
        features["zero_crossing_rate"] = np.nan
    
    # Extract MFCCs
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(min(20, mfccs.shape[0])):
            features[f"mfcc{i+1}"] = mfccs[i].mean()
    except:
        for i in range(20):
            features[f"mfcc{i+1}"] = np.nan
    
    return features

# Main processing function
def process_audio_files(input_folder, output_csv="features.csv"):
    """Process audio files and extract features."""
    if not os.path.exists(input_folder):
        print(f"❌ Folder '{input_folder}' not found.")
        return False
    
    # Check if folder has files
    if not os.listdir(input_folder):
        print(f"❌ Folder '{input_folder}' is empty.")
        return False
    
    # Check if ffmpeg is installed
    ffmpeg_path = check_ffmpeg()
    if not ffmpeg_path:
        print("❌ FFmpeg not found. Please install FFmpeg and make sure it's in your PATH.")
        print("Download FFmpeg from: https://ffmpeg.org/download.html")
        return False
    
    # Get list of audio files - include MP3 files too
    audio_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith((".wav", ".mp3"))]
    
    if not audio_files:
        print(f"❌ No audio files found in '{input_folder}'.")
        print(f"Files in directory: {os.listdir(input_folder)}")
        return False
    
    print(f"🎵 Found {len(audio_files)} audio files to process.")
    print(f"First few files: {audio_files[:5]}")
    
    # Diagnostic check on first file
    first_file = os.path.join(input_folder, audio_files[0])
    print(f"\n===== DIAGNOSING FIRST FILE =====")
    is_valid = check_audio_file(first_file)
    print("============================\n")
    
    # Always convert files to ensure consistent format
    print("Converting all files to standard WAV format...")
    
    # Create a temp directory for converted files
    temp_dir = "converted_wav"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert all files
    converted_files = []
    for filename in tqdm(audio_files, desc="Converting files"):
        input_path = os.path.join(input_folder, filename)
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            continue
            
        output_path = convert_to_standard_wav(input_path, temp_dir, ffmpeg_path)
        if output_path:
            converted_files.append((filename, output_path))
    
    if not converted_files:
        print("❌ Failed to convert any files. Please check file formats and ffmpeg installation.")
        return False
        
    print(f"✅ Successfully converted {len(converted_files)} files.")
    
    # Use converted files for feature extraction
    all_features = []
    
    for original_name, filepath in tqdm(converted_files, desc="Extracting features"):
        try:
            # Load audio with conservative settings
            y, sr = librosa.load(filepath, sr=22050, mono=True)
            
            # Extract features
            features = extract_features(y, sr)
            features["filename"] = original_name  # Keep original filename
            
            all_features.append(features)
            
        except Exception as e:
            print(f"❌ Error processing {original_name}: {str(e)}")
    
    # Save results
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Reorder columns to put filename first
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("filename")))
        df = df[cols]
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Features extracted from {len(all_features)} files and saved to {output_csv}")
        
        # Print summary
        print(f"\n📊 Features extracted: {len(df.columns)-1} features for {len(df)} files")
        return True
    else:
        print("⚠️ No valid features were extracted from any files.")
        return False

if __name__ == "__main__":
    # Set folder path
    input_folder = "music2"  # Change to your folder path
    output_csv = "features.csv"
    
    # Print debugging info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input folder: {input_folder}")
    print(f"Input folder full path: {os.path.abspath(input_folder)}")
    if os.path.exists(input_folder):
        print(f"Input folder exists: Yes")
        print(f"Files in input folder: {os.listdir(input_folder)}")
    else:
        print(f"Input folder exists: No")
    
    # Run the processing
    process_audio_files(input_folder, output_csv)
