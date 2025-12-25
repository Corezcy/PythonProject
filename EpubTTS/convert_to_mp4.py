#!/usr/bin/env python3

"""
Batch convert MP3+SRT files to MP4 with minimal video quality
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Thread-safe print lock
print_lock = Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def convert_to_mp4(mp3_file: str, srt_file: str, mp4_file: str) -> tuple[bool, str]:
    """Convert MP3 + SRT to MP4 with minimal video quality using ffmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-f', 'lavfi', '-i', 'color=c=black:s=320x240:r=1',  # 4:3 aspect ratio
            '-i', mp3_file,
            '-i', srt_file,
            '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '51',
            '-c:a', 'copy',
            '-c:s', 'mov_text',
            '-shortest',
            '-y',
            mp4_file
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            return (True, f"✓ {os.path.basename(mp4_file)}")
        else:
            return (False, f"✗ {os.path.basename(mp4_file)}: {result.stderr[:100]}")
    except Exception as e:
        return (False, f"✗ {os.path.basename(mp4_file)}: {e}")

def process_file(mp3_file: str, output_dir: str, total: int, index: int) -> tuple[bool, str]:
    """Process a single MP3+SRT pair."""
    base_name = os.path.splitext(os.path.basename(mp3_file))[0]
    srt_file = os.path.join(output_dir, f"{base_name}.srt")
    mp4_file = os.path.join(output_dir, f"{base_name}.mp4")
    
    if not os.path.exists(srt_file):
        return (False, f"✗ SRT not found for {base_name}")
    
    thread_safe_print(f"[{index}/{total}] Converting {base_name}...")
    success, message = convert_to_mp4(mp3_file, srt_file, mp4_file)
    thread_safe_print(f"  {message}")
    
    return (success, message)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_to_mp4.py <directory> [max_threads]")
        print("\nExample:")
        print("  python convert_to_mp4.py tts_output")
        print("  python convert_to_mp4.py tts_output 4")
        sys.exit(1)
    
    directory = sys.argv[1]
    max_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found!")
        sys.exit(1)
    
    # Find all MP3 files
    mp3_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')])
    
    if not mp3_files:
        print(f"No MP3 files found in {directory}")
        sys.exit(1)
    
    total = len(mp3_files)
    print(f"Found {total} MP3 files")
    print(f"Converting with {max_threads} threads\n")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_file, mp3_file, directory, total, i+1): i 
            for i, mp3_file in enumerate(mp3_files)
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception: {e}")
                results.append((False, str(e)))
    
    success_count = sum(1 for r in results if r[0])
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{total} files converted successfully")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
