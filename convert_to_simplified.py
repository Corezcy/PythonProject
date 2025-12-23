#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具脚本：批量转换当前目录下所有 *_transcript.txt 文件为简体中文

用法:
  python convert_to_simplified.py

依赖:
  pip install opencc
"""

import os
import glob
from opencc import OpenCC


def convert_file_to_simplified(input_file, output_file):
    """将单个文件从繁体转换为简体中文"""
    cc = OpenCC('t2s')  # 繁体转简体转换器
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            simplified_line = cc.convert(line)
            outfile.write(simplified_line)


def batch_convert_transcripts():
    """批量转换当前目录下所有 *_transcript.txt 文件"""
    # 查找所有符合条件的文件
    pattern = "*_transcript.txt"
    transcript_files = glob.glob(pattern)
    
    if not transcript_files:
        print(f"未找到任何匹配 '{pattern}' 的文件")
        return
    
    print(f"找到 {len(transcript_files)} 个文件待转换:")
    for file in transcript_files:
        print(f"  - {file}")
    
    print("\n开始转换...")
    
    converted_count = 0
    for input_file in transcript_files:
        # 生成输出文件名：在 _transcript.txt 前插入 _simplified
        # 例如: test_transcript.txt -> test_transcript_simplified.txt
        base_name = input_file.replace('_transcript.txt', '')
        output_file = f"{base_name}_transcript_simplified.txt"
        
        try:
            convert_file_to_simplified(input_file, output_file)
            print(f"✓ {input_file} -> {output_file}")
            converted_count += 1
        except Exception as e:
            print(f"✗ 转换失败 {input_file}: {e}")
    
    print(f"\n转换完成! 成功转换 {converted_count}/{len(transcript_files)} 个文件")


if __name__ == "__main__":
    batch_convert_transcripts()
