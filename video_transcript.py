#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频音轨提取与字幕生成工具 (跨平台加速版)
支持: macOS (Metal), Linux (CUDA), Windows (CPU)
需要安装: pip install faster-whisper ffmpeg-python opencc
系统需要安装 FFmpeg
"""

import os
import sys
import platform
from faster_whisper import WhisperModel
import ffmpeg
from datetime import timedelta
from tqdm import tqdm
import time
from opencc import OpenCC

cc_converter = OpenCC('t2s')  # 繁体转简体转换器


def detect_device():
    """
    自动检测最佳计算设备
    
    Returns:
        tuple: (device, compute_type, description)
    """
    system = platform.system()
    
    # 尝试检测 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda", "float16", f"NVIDIA GPU: {gpu_name}"
    except ImportError:
        pass
    
    # macOS 使用 CPU (自动利用 Metal)
    if system == "Darwin":
        return "cpu", "int8", "macOS CPU (Metal 加速)"
    
    # Linux/Windows 没有 CUDA 则使用 CPU
    return "cpu", "int8", f"{system} CPU"


def format_timestamp(seconds):
    """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_video_duration(video_path):
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        float: 视频时长（秒）
    """
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        print(f"警告: 无法获取视频时长: {e}")
        return None


def extract_audio(video_path, audio_path):
    """
    从视频文件中提取音轨
    
    Args:
        video_path: 视频文件路径
        audio_path: 输出音频文件路径
    """
    print(f"正在从 {video_path} 提取音轨...")
    
    try:
        # 使用 ffmpeg-python 提取音频
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, 
                              acodec='pcm_s16le',  # WAV 格式
                              ac=1,                 # 单声道
                              ar='16000')           # 采样率 16kHz
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"✓ 音轨提取完成: {audio_path}")
        return True
    except ffmpeg.Error as e:
        print(f"✗ 提取音轨失败: {e.stderr.decode()}")
        return False


def transcribe_audio_fast(audio_path, model_size="medium", language="zh", device="auto", duration=None):
    """
    使用 faster-whisper 进行语音识别
    自动选择最佳设备 (CUDA/Metal/CPU)
    
    Args:
        audio_path: 音频文件路径
        model_size: 模型大小 (tiny, base, small, medium, large-v2, large-v3)
        language: 语言代码 (zh=中文, en=英文, auto=自动检测)
        device: 计算设备 (auto, cpu, cuda)
        duration: 音频时长（秒），用于显示进度
    
    Returns:
        转录结果列表
    """
    # 自动检测设备
    if device == "auto":
        device, compute_type, device_desc = detect_device()
        print(f"\n检测到计算设备: {device_desc}")
    else:
        # 手动指定设备
        if device == "cuda":
            compute_type = "float16"
            device_desc = "NVIDIA CUDA GPU"
        else:
            compute_type = "int8"
            device_desc = "CPU"
        print(f"\n使用指定设备: {device_desc}")
    
    print(f"正在加载 faster-whisper {model_size} 模型...")
    print(f"计算精度: {compute_type}")
    
    # 根据设备配置参数
    model_kwargs = {
        "model_size_or_path": model_size,
        "device": device,
        "compute_type": compute_type
    }
    
    # CPU 模式下设置线程数
    if device == "cpu":
        model_kwargs["cpu_threads"] = os.cpu_count() or 4
    
    model = WhisperModel(**model_kwargs)
    
    print(f"正在识别语音 (语言: {language})...")
    if duration:
        print(f"音频时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    
    # 配置转录参数
    transcribe_kwargs = {
        "audio": audio_path,
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": dict(min_silence_duration_ms=500)
    }
    
    # 执行转录
    start_time = time.time()
    
    if language.lower() == "auto":
        segments, info = model.transcribe(**transcribe_kwargs)
        detected_language = info.language
        print(f"检测到的语言: {detected_language} (置信度: {info.language_probability:.2%})")
    else:
        transcribe_kwargs["language"] = language
        segments, info = model.transcribe(**transcribe_kwargs)
    
    # 使用进度条收集片段
    segments_list = []
    
    if duration:
        # 有时长信息，显示百分比进度
        with tqdm(total=100, desc="识别进度", unit="%", ncols=80, 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            last_progress = 0
            for segment in segments:
                segments_list.append(segment)
                # 根据时间计算进度
                current_progress = int((segment.end / duration) * 100)
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                    # 显示当前处理的时间点
                    pbar.set_postfix_str(f"时间: {segment.end:.1f}s / {duration:.1f}s")
            # 确保进度条到达 100%
            pbar.update(100 - last_progress)
    else:
        # 没有时长信息，只显示片段计数
        print("\n开始识别...")
        with tqdm(desc="处理片段", unit="段", ncols=80) as pbar:
            for segment in segments:
                segments_list.append(segment)
                pbar.update(1)
                pbar.set_postfix_str(f"当前时间: {segment.end:.1f}s")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ 语音识别完成 (共 {len(segments_list)} 个片段)")
    print(f"识别耗时: {elapsed_time:.1f} 秒")
    
    # 计算识别速度
    if duration and duration > 0:
        speed_ratio = duration / elapsed_time
        print(f"处理速度: {speed_ratio:.2f}x 实时速度")
        if device == "cuda":
            print(f"GPU 加速效果显著 🚀")
    
    return segments_list


def save_srt(segments, srt_path):
    """
    将识别结果保存为 SRT 字幕文件，自动将繁体转换为简体
    
    Args:
        segments: faster-whisper 识别的片段列表
        srt_path: 输出 SRT 文件路径
    """
    print(f"\n正在生成 SRT 字幕文件...")
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            # 字幕序号
            f.write(f"{i}\n")
            
            # 时间轴
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            f.write(f"{start_time} --> {end_time}\n")
            
            # 字幕内容，转换为简体中文
            simplified_text = cc_converter.convert(segment.text.strip())
            f.write(f"{simplified_text}\n\n")
    
    print(f"✓ SRT 字幕已保存: {srt_path}")


def save_txt(segments, txt_path):
    """
    将识别结果保存为纯文本文件
    
    Args:
        segments: faster-whisper 识别的片段列表
        txt_path: 输出 TXT 文件路径
    """
    print(f"正在生成文本文件...")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")
    
    print(f"✓ 文本已保存: {txt_path}")


def print_system_info():
    """打印系统和环境信息"""
    print("\n系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CPU 核心数: {os.cpu_count()}")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"  CUDA: 不可用")
    except ImportError:
        print(f"  PyTorch: 未安装")


def main():
    """主函数"""
    # 配置参数
    if len(sys.argv) < 2:
        print("=" * 70)
        print("视频字幕生成工具 (跨平台加速版)")
        print("=" * 70)
        print("\n用法: python script.py <视频文件路径> [模型大小] [语言] [设备]")
        print("\n示例:")
        print("  python script.py video.mp4                    # 自动检测设备")
        print("  python script.py video.mp4 medium zh          # 指定模型和语言")
        print("  python script.py video.mp4 medium zh cuda     # 强制使用 CUDA")
        print("  python script.py video.mp4 small en cpu       # 强制使用 CPU")
        print("\n参数说明:")
        print("  模型大小: tiny, base, small, medium, large-v2, large-v3")
        print("    - tiny/base: 极快，适合实时字幕 (GPU: <1秒/分钟)")
        print("    - small: 快速且准确度不错 (GPU: ~2秒/分钟)")
        print("    - medium: 推荐，准确度高 (GPU: ~5秒/分钟, CPU: ~2.5分钟/分钟)")
        print("    - large-v3: 最高准确度 (GPU: ~10秒/分钟, CPU: ~6分钟/分钟)")
        print("\n  语言: zh(中文), en(英文), ja(日语), auto(自动检测)")
        print("\n  设备: auto(自动), cuda(GPU), cpu")
        print("\n平台优化:")
        print("  ✓ Linux + CUDA: 自动使用 float16 精度，速度最快")
        print("  ✓ macOS: 自动使用 Metal 加速")
        print("  ✓ Windows/Linux(无GPU): 使用 int8 量化加速")
        
        print_system_info()
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    language = sys.argv[3] if len(sys.argv) > 3 else "zh"
    device = sys.argv[4] if len(sys.argv) > 4 else "auto"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 找不到文件 {video_path}")
        sys.exit(1)
    
    # 生成输出文件路径
    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}_audio.wav"
    srt_path = f"{base_name}.srt"
    txt_path = f"{base_name}_transcript.txt"
    
    print("=" * 70)
    print("视频字幕生成工具 (跨平台加速版)")
    print("=" * 70)
    print(f"输入视频: {video_path}")
    print(f"模型大小: {model_size}")
    print(f"识别语言: {language}")
    print(f"计算设备: {device}")
    print("=" * 70)
    
    # 获取视频时长
    duration = get_video_duration(video_path)
    if duration:
        print(f"视频时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    
    # 步骤 1: 提取音轨
    if not extract_audio(video_path, audio_path):
        sys.exit(1)
    
    # 步骤 2: 语音识别
    try:
        segments = transcribe_audio_fast(audio_path, model_size, language, device, duration)
    except Exception as e:
        print(f"\n✗ 语音识别失败: {e}")
        print("\n故障排除:")
        print("1. 确保已安装: pip install faster-whisper ffmpeg-python tqdm")
        print("2. 确保已安装 FFmpeg")
        print("   - macOS: brew install ffmpeg")
        print("   - Linux: sudo apt install ffmpeg")
        print("   - Windows: 从 https://ffmpeg.org 下载")
        print("3. CUDA 用户需要安装: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    # 步骤 3: 保存字幕文件
    save_srt(segments, srt_path)
    save_txt(segments, txt_path)
    
    # 清理临时音频文件（可选）
    cleanup = input("\n是否删除临时音频文件? (y/n): ")
    if cleanup.lower() == 'y':
        os.remove(audio_path)
        print(f"✓ 已删除: {audio_path}")
    
    print("\n" + "=" * 70)
    print("✓ 全部完成!")
    print("=" * 70)
    print(f"字幕文件: {srt_path}")
    print(f"文本文件: {txt_path}")


if __name__ == "__main__":
    main()
