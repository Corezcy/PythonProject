#!/usr/bin/env python3

"""
EPUB to TTS Converter
Reads EPUB books and converts content to speech using edge-tts,
splitting content into 10,000 character chunks for processing.
Uses multi-threading for concurrent processing.
"""

import asyncio
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import edge_tts

# Configuration
VOICE = "zh-CN-XiaoxiaoNeural"  # Chinese female voice, change as needed
CHUNK_SIZE = 10000  # 10,000 characters per chunk
OUTPUT_DIR = "tts_output"
MAX_THREADS = 8  # Maximum number of concurrent threads

# Thread-safe print lock
print_lock = Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)


def extract_text_from_epub(epub_path: str) -> str:
    """
    Extract all text content from an EPUB file.
    
    Args:
        epub_path: Path to the EPUB file
        
    Returns:
        Extracted text content as a single string
    """
    book = epub.read_epub(epub_path)
    text_content = []
    
    # Iterate through all items in the EPUB
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Extract text and remove extra whitespace
            text = soup.get_text(separator=' ', strip=True)
            if text:
                text_content.append(text)
    
    # Join all text with newlines
    full_text = '\n\n'.join(text_content)
    return full_text


def extract_chapters_from_epub(epub_path: str) -> list[tuple[str, str]]:
    """
    Extract chapters from EPUB file as separate chunks.
    
    Args:
        epub_path: Path to the EPUB file
        
    Returns:
        List of tuples (chapter_title, chapter_text)
    """
    book = epub.read_epub(epub_path)
    chapters = []
    
    # Iterate through all items in the EPUB
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Try to extract chapter title from h1, h2, or title tags
            title = None
            for tag in ['h1', 'h2', 'h3', 'title']:
                title_tag = soup.find(tag)
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    break
            
            # If no title found, use filename
            if not title:
                title = item.get_name().split('/')[-1].replace('.xhtml', '').replace('.html', '')
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            if text and len(text) > 100:  # Only include substantial chapters
                chapters.append((title, text))
    
    return chapters


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks of specified size.
    Try to break at sentence boundaries when possible.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # Calculate end position
        end_pos = min(current_pos + chunk_size, text_length)
        
        # If not at the end, try to find a good breaking point
        if end_pos < text_length:
            # Look for sentence endings (。！？.!?) within the last 200 characters
            search_start = max(current_pos, end_pos - 200)
            chunk_text = text[search_start:end_pos]
            
            # Try to find Chinese sentence endings first
            for delimiter in ['。', '！', '？', '.\n', '!\n', '?\n', '. ', '! ', '? ']:
                last_occurrence = chunk_text.rfind(delimiter)
                if last_occurrence != -1:
                    end_pos = search_start + last_occurrence + len(delimiter)
                    break
        
        # Extract chunk
        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        current_pos = end_pos
    
    return chunks


def split_chapters_by_size(chapters: list[tuple[str, str]], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split chapters into smaller chunks if they exceed chunk_size.
    
    Args:
        chapters: List of (title, text) tuples
        chunk_size: Maximum size for each chunk
        
    Returns:
        List of text chunks
    """
    final_chunks = []
    
    for title, text in chapters:
        chapter_length = len(text)
        
        # If chapter is small enough, keep it as one chunk
        if chapter_length <= chunk_size:
            final_chunks.append(text)
        else:
            # Split large chapter into smaller chunks
            sub_chunks = split_text_into_chunks(text, chunk_size)
            final_chunks.extend(sub_chunks)
    
    return final_chunks


def text_to_speech_sync(text: str, output_file: str, voice: str = VOICE, 
                        vtt_file: str = None) -> None:
    """
    Convert text to speech using edge-tts and save to file.
    This is a synchronous wrapper around the async function.
    
    Args:
        text: Text to convert
        output_file: Output audio file path
        voice: Voice to use for TTS
        vtt_file: Optional VTT subtitle file path
    """
    async def _async_tts():
        communicate = edge_tts.Communicate(text, voice)
        submaker = edge_tts.SubMaker()
        
        with open(output_file, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    submaker.feed(chunk)
        
        # Save VTT subtitle if requested
        if vtt_file:
            with open(vtt_file, "w", encoding="utf-8") as subtitle_file:
                subtitle_file.write(submaker.get_srt())
    
    # Run the async function in a new event loop
    asyncio.run(_async_tts())


def process_chunk(chunk_data: tuple, epub_filename: str, output_dir: str, 
                 voice: str, total_chunks: int, generate_subtitles: bool = False) -> tuple[int, bool, str]:
    """
    Process a single chunk in a thread.
    
    Args:
        chunk_data: Tuple of (chunk_index, chunk_text)
        epub_filename: Base filename for output
        output_dir: Directory to save output files
        voice: Voice to use for TTS
        total_chunks: Total number of chunks
        generate_subtitles: Whether to generate VTT subtitles
        
    Returns:
        Tuple of (chunk_num, success, message)
    """
    i, chunk = chunk_data
    output_file = os.path.join(output_dir, f"{epub_filename}_part_{i:03d}.mp3")
    vtt_file = os.path.join(output_dir, f"{epub_filename}_part_{i:03d}.vtt") if generate_subtitles else None
    
    thread_safe_print(f"[Thread] Processing chunk {i}/{total_chunks} ({len(chunk)} chars) -> {output_file}")
    
    try:
        text_to_speech_sync(chunk, output_file, voice, vtt_file)
        message = f"✓ Successfully created {output_file}"
        if vtt_file:
            message += f" + {os.path.basename(vtt_file)}"
        thread_safe_print(f"  {message}")
        return (i, True, message)
    except Exception as e:
        message = f"✗ Error processing chunk {i}: {e}"
        thread_safe_print(f"  {message}")
        return (i, False, message)


def convert_to_mp4(mp3_file: str, vtt_file: str, mp4_file: str) -> tuple[bool, str]:
    """
    Convert MP3 + VTT to MP4 with minimal video quality using ffmpeg.
    
    Args:
        mp3_file: Path to MP3 audio file
        vtt_file: Path to VTT subtitle file
        mp4_file: Path to output MP4 file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # FFmpeg command with minimal video quality
        cmd = [
            'ffmpeg',
            '-f', 'lavfi', '-i', 'color=c=black:s=16x16:r=1',  # Minimal black video
            '-i', mp3_file,  # Audio input
            '-i', vtt_file,  # Subtitle input
            '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0',  # Map video, audio, subtitle
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '51',  # Minimal video quality
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-c:s', 'mov_text',  # Subtitle codec for MP4
            '-shortest',  # Match duration to shortest input
            '-y',  # Overwrite output file
            mp4_file
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            return (True, f"✓ Successfully created {os.path.basename(mp4_file)}")
        else:
            return (False, f"✗ FFmpeg error: {result.stderr[:200]}")
            
    except FileNotFoundError:
        return (False, "✗ FFmpeg not found. Please install ffmpeg.")
    except Exception as e:
        return (False, f"✗ Error: {e}")


def convert_chunk_to_mp4(chunk_num: int, epub_filename: str, output_dir: str, 
                        total_chunks: int) -> tuple[int, bool, str]:
    """
    Convert a single chunk's MP3+VTT to MP4 in a thread.
    
    Args:
        chunk_num: Chunk number
        epub_filename: Base filename
        output_dir: Output directory
        total_chunks: Total number of chunks
        
    Returns:
        Tuple of (chunk_num, success, message)
    """
    mp3_file = os.path.join(output_dir, f"{epub_filename}_part_{chunk_num:03d}.mp3")
    vtt_file = os.path.join(output_dir, f"{epub_filename}_part_{chunk_num:03d}.vtt")
    mp4_file = os.path.join(output_dir, f"{epub_filename}_part_{chunk_num:03d}.mp4")
    
    # Check if files exist
    if not os.path.exists(mp3_file):
        return (chunk_num, False, f"✗ MP3 file not found: {mp3_file}")
    if not os.path.exists(vtt_file):
        return (chunk_num, False, f"✗ VTT file not found: {vtt_file}")
    
    thread_safe_print(f"[FFmpeg] Converting chunk {chunk_num}/{total_chunks} to MP4...")
    
    success, message = convert_to_mp4(mp3_file, vtt_file, mp4_file)
    thread_safe_print(f"  {message}")
    
    return (chunk_num, success, message)


def process_epub_to_tts(epub_path: str, output_dir: str = OUTPUT_DIR, 
                        voice: str = VOICE, max_threads: int = MAX_THREADS,
                        generate_subtitles: bool = False, convert_to_mp4_flag: bool = False,
                        force_split: bool = False, chunk_size: int = CHUNK_SIZE) -> None:
    """
    Main function to process EPUB file and convert to TTS audio files.
    Uses ThreadPoolExecutor for concurrent processing.
    
    Args:
        epub_path: Path to the EPUB file
        output_dir: Directory to save output audio files
        voice: Voice to use for TTS
        max_threads: Maximum number of concurrent threads
        generate_subtitles: Whether to generate VTT subtitle files
        convert_to_mp4_flag: Whether to convert MP3+VTT to MP4
        force_split: Force split by character count instead of chapters
        chunk_size: Maximum characters per chunk when splitting
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract base filename without extension
    epub_filename = Path(epub_path).stem
    
    print(f"Reading EPUB file: {epub_path}")
    
    # Determine splitting strategy
    if force_split:
        # Force split by character count
        print("Split mode: By character count (forced)")
        full_text = extract_text_from_epub(epub_path)
        print(f"Extracted {len(full_text)} characters from EPUB")
        chunks = split_text_into_chunks(full_text, chunk_size)
    else:
        # Default: Split by chapters
        print("Split mode: By chapters (default)")
        chapters = extract_chapters_from_epub(epub_path)
        print(f"Found {len(chapters)} chapters in EPUB")
        
        # Show chapter info
        total_chars = sum(len(text) for _, text in chapters)
        print(f"Total characters: {total_chars}")
        
        # Split chapters by size if needed
        chunks = split_chapters_by_size(chapters, chunk_size)
    
    total_chunks = len(chunks)
    print(f"Split into {total_chunks} chunks")
    print(f"Processing with {max_threads} concurrent threads\n")
    
    # Create chunk data
    chunk_data = [(i, chunk) for i, chunk in enumerate(chunks, 1)]
    
    # Process chunks using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk, data, epub_filename, output_dir, voice, total_chunks, generate_subtitles): data[0]
            for data in chunk_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                chunk_num = future_to_chunk[future]
                thread_safe_print(f"  ✗ Exception in thread for chunk {chunk_num}: {e}")
                results.append((chunk_num, False, str(e)))
    
    # Count successes and failures
    success_count = sum(1 for r in results if r[1])
    failure_count = total_chunks - success_count
    
    print(f"\n{'='*60}")
    print(f"Completed! Generated {success_count}/{total_chunks} audio files successfully")
    if failure_count > 0:
        print(f"Failed: {failure_count} chunks")
        failed_chunks = sorted([r[0] for r in results if not r[1]])
        print(f"Failed chunk numbers: {failed_chunks}")
    print(f"Output directory: '{output_dir}'")
    print(f"{'='*60}")
    
    # Convert to MP4 if requested
    if convert_to_mp4_flag and generate_subtitles and success_count > 0:
        print(f"\n{'='*60}")
        print(f"Converting {success_count} audio+subtitle pairs to MP4...")
        print(f"{'='*60}\n")
        
        # Get successful chunk numbers
        successful_chunks = [r[0] for r in results if r[1]]
        
        # Convert using ThreadPoolExecutor
        mp4_results = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_chunk = {
                executor.submit(convert_chunk_to_mp4, chunk_num, epub_filename, output_dir, total_chunks): chunk_num
                for chunk_num in successful_chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    mp4_results.append(result)
                except Exception as e:
                    chunk_num = future_to_chunk[future]
                    thread_safe_print(f"  ✗ Exception converting chunk {chunk_num}: {e}")
                    mp4_results.append((chunk_num, False, str(e)))
        
        # Count MP4 conversion successes
        mp4_success_count = sum(1 for r in mp4_results if r[1])
        mp4_failure_count = len(mp4_results) - mp4_success_count
        
        print(f"\n{'='*60}")
        print(f"MP4 Conversion: {mp4_success_count}/{len(mp4_results)} files converted successfully")
        if mp4_failure_count > 0:
            print(f"Failed: {mp4_failure_count} conversions")
            failed_mp4_chunks = sorted([r[0] for r in mp4_results if not r[1]])
            print(f"Failed chunk numbers: {failed_mp4_chunks}")
        print(f"{'='*60}")
    elif convert_to_mp4_flag and not generate_subtitles:
        print("\n⚠ Warning: --mp4 flag requires --subtitles flag. Skipping MP4 conversion.")


def main() -> None:
    """Main entry point"""
    import sys
    
    # Check if EPUB file path is provided
    if len(sys.argv) < 2:
        print("Usage: python epub_to_tts.py <epub_file_path> [voice] [output_dir] [max_threads] [--subtitles] [--mp4] [--force-split] [--chunk-size SIZE]")
        print("\nExamples:")
        print("  python epub_to_tts.py book.epub")
        print("  python epub_to_tts.py book.epub --subtitles --mp4")
        print("  python epub_to_tts.py book.epub --force-split")
        print("  python epub_to_tts.py book.epub --force-split --chunk-size 5000")
        print("  python epub_to_tts.py book.epub zh-CN-YunxiNeural ./output 10 --subtitles")
        print("\nCommon voices:")
        print("  Chinese: zh-CN-XiaoxiaoNeural (female), zh-CN-YunxiNeural (male)")
        print("  English: en-US-JennyNeural (female), en-US-GuyNeural (male)")
        print("  Use 'edge-tts --list-voices' to see all available voices")
        print(f"\nDefault concurrent threads: {MAX_THREADS}")
        print(f"Default chunk size: {CHUNK_SIZE} characters")
        print("\nOptions:")
        print("  --subtitles       Generate VTT subtitle files for each audio chunk")
        print("  --mp4             Convert MP3+VTT to MP4 with minimal video (requires --subtitles)")
        print("  --force-split     Force split by character count instead of chapters (default: by chapters)")
        print("  --chunk-size N    Set chunk size to N characters (default: 10000)")
        sys.exit(1)
    
    epub_path = sys.argv[1]
    
    # Parse flags
    generate_subtitles = '--subtitles' in sys.argv
    convert_to_mp4_flag = '--mp4' in sys.argv
    force_split = '--force-split' in sys.argv
    
    # Parse chunk size
    chunk_size = CHUNK_SIZE
    if '--chunk-size' in sys.argv:
        try:
            chunk_size_idx = sys.argv.index('--chunk-size')
            if chunk_size_idx + 1 < len(sys.argv):
                chunk_size = int(sys.argv[chunk_size_idx + 1])
        except (ValueError, IndexError):
            print("Error: Invalid --chunk-size value")
            sys.exit(1)
    
    # Remove flags from argv for easier parsing
    args = [arg for arg in sys.argv[1:] if arg not in ['--subtitles', '--mp4', '--force-split', '--chunk-size'] and not arg.isdigit() or arg == sys.argv[1]]
    # Remove chunk size value if present
    if '--chunk-size' in sys.argv:
        chunk_size_idx = sys.argv.index('--chunk-size')
        if chunk_size_idx + 1 < len(sys.argv) and sys.argv[chunk_size_idx + 1].isdigit():
            chunk_size_val = sys.argv[chunk_size_idx + 1]
            if chunk_size_val in args:
                args.remove(chunk_size_val)
    
    voice = args[1] if len(args) > 1 else VOICE
    output_dir = args[2] if len(args) > 2 else OUTPUT_DIR
    max_threads = int(args[3]) if len(args) > 3 else MAX_THREADS
    
    # Check if file exists
    if not os.path.exists(epub_path):
        print(f"Error: File '{epub_path}' not found!")
        sys.exit(1)
    
    # Show configuration
    print(f"Configuration:")
    print(f"  Voice: {voice}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max threads: {max_threads}")
    print(f"  Chunk size: {chunk_size} characters")
    print(f"  Split mode: {'By character count' if force_split else 'By chapters'}")
    print(f"  Generate subtitles: {'Yes' if generate_subtitles else 'No'}")
    print(f"  Convert to MP4: {'Yes' if convert_to_mp4_flag else 'No'}")
    print()
    
    # Process the EPUB file
    process_epub_to_tts(epub_path, output_dir, voice, max_threads, generate_subtitles, convert_to_mp4_flag, force_split, chunk_size)


"""
使用说明 / Usage Instructions:

基本用法 / Basic Usage:
    python epub_to_tts.py book.epub

完整功能 / Full Features:
    python epub_to_tts.py book.epub --subtitles --mp4

拆分模式 / Split Modes:
    # 默认：按章节拆分 (推荐)
    python epub_to_tts.py book.epub
    
    # 强制按字数拆分
    python epub_to_tts.py book.epub --force-split
    
    # 自定义分块大小 (5000 字符)
    python epub_to_tts.py book.epub --force-split --chunk-size 5000

高级选项 / Advanced Options:
    python epub_to_tts.py book.epub zh-CN-YunxiNeural ./output 10 --subtitles --mp4
    # book.epub: EPUB 文件路径
    # zh-CN-YunxiNeural: 语音选择 (男声)
    # ./output: 输出目录
    # 10: 线程数
    # --subtitles: 生成 VTT 字幕
    # --mp4: 转换为 MP4 视频

常用语音 / Common Voices:
    中文: zh-CN-XiaoxiaoNeural (女), zh-CN-YunxiNeural (男)
    英文: en-US-JennyNeural (女), en-US-GuyNeural (男)
    查看所有: edge-tts --list-voices

输出文件 / Output Files:
    - book_part_001.mp3  # 音频文件
    - book_part_001.vtt  # 字幕文件 (使用 --subtitles)
    - book_part_001.mp4  # 视频文件 (使用 --mp4)

工具脚本 / Helper Scripts:
    # 批量转换已有的 MP3+VTT 为 MP4
    python convert_to_mp4.py tts_output 8

依赖安装 / Install Dependencies:
    pip install -r requirements_epub_tts.txt
"""

if __name__ == "__main__":
    main()
