#!/usr/bin/env python3
"""
脚本：将CSV文件中的特征路径列替换为wav文件路径列

对于目录和子目录的所有csv文件，将prosody,wav2vec,phonation,articulation,glottal列
替换为一个wav_path列，其中路径格式为：
'/data/storage1t/data_nofolder/gita/norm_audios/' + filename.wav

作者：自动生成
日期：2025-08-15
"""

import pandas as pd
import os
import glob
from pathlib import Path
import argparse

def extract_filename_from_path(feature_path):
    """
    从特征文件路径中提取文件名并转换为wav文件名
    
    Args:
        feature_path (str): 特征文件路径，例如 './data/gita/speech_features/disvoice/prosody/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz'
    
    Returns:
        str: wav文件名，例如 'PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.wav'
    """
    if pd.isna(feature_path) or feature_path == '':
        return ''
    
    # 分割路径并获取文件名部分
    path_parts = feature_path.split('/')
    if len(path_parts) >= 2:
        filename = path_parts[-1]  # 获取最后一个部分（文件名）
        # 将.npz替换为.wav
        wav_filename = filename.replace('.npz', '.wav')
        return wav_filename
    else:
        # 如果路径格式不符合预期，直接处理文件名部分
        filename = os.path.basename(feature_path)
        wav_filename = filename.replace('.npz', '.wav')
        return wav_filename

def create_wav_path(feature_path, base_path='/data/storage1t/data_nofolder/gita/norm_audios/'):
    """
    根据特征文件路径创建wav文件的完整路径
    
    Args:
        feature_path (str): 原始特征文件路径
        base_path (str): wav文件的基础路径
    
    Returns:
        str: 完整的wav文件路径
    """
    wav_filename = extract_filename_from_path(feature_path)
    if wav_filename:
        return base_path + wav_filename
    else:
        return ''

def process_csv_file(csv_file_path, base_wav_path='/data/storage1t/data_nofolder/gita/norm_audios/', backup=True):
    """
    处理单个CSV文件，替换特征列为wav_path列
    
    Args:
        csv_file_path (str): CSV文件路径
        base_wav_path (str): wav文件基础路径
        backup (bool): 是否创建备份文件
    """
    try:
        print(f"处理文件: {csv_file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 定义要替换的列
        feature_columns = ['prosody', 'wav2vec', 'phonation', 'articulation', 'glottal']
        
        # 检查哪些列存在
        existing_columns = [col for col in feature_columns if col in df.columns]
        
        if not existing_columns:
            print(f"  警告: 文件 {csv_file_path} 中没有找到特征列，跳过处理")
            return
        
        print(f"  找到特征列: {existing_columns}")
        
        # 创建备份（如果需要）
        if backup:
            backup_path = csv_file_path + '.backup'
            if not os.path.exists(backup_path):
                df.to_csv(backup_path, index=False)
                print(f"  创建备份文件: {backup_path}")
        
        # 使用prosody列（如果存在）或第一个存在的特征列来创建wav_path
        reference_column = 'prosody' if 'prosody' in existing_columns else existing_columns[0]
        
        # 创建wav_path列
        df['wav_path'] = df[reference_column].apply(lambda x: create_wav_path(x, base_wav_path))
        
        # 删除原有的特征列
        df = df.drop(columns=existing_columns)
        
        # 保存修改后的文件
        df.to_csv(csv_file_path, index=False)
        
        print(f"  成功处理，删除了 {len(existing_columns)} 个特征列，添加了 wav_path 列")
        print(f"  总共 {len(df)} 行数据")
        
    except Exception as e:
        print(f"  错误: 处理文件 {csv_file_path} 时出现错误: {str(e)}")

def find_csv_files(root_directory):
    """
    在指定目录及其子目录中查找所有CSV文件
    
    Args:
        root_directory (str): 根目录路径
    
    Returns:
        list: CSV文件路径列表
    """
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def main():
    parser = argparse.ArgumentParser(description='将CSV文件中的特征路径列替换为wav文件路径列')
    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('--wav-base-path', default='/data/storage1t/data_nofolder/gita/norm_audios/',
                        help='wav文件的基础路径 (默认: /data/storage1t/data_nofolder/gita/norm_audios/)')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份文件')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要处理的文件，不实际修改')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"错误: 目录 {args.directory} 不存在")
        return
    
    # 查找所有CSV文件
    csv_files = find_csv_files(args.directory)
    
    if not csv_files:
        print(f"在目录 {args.directory} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for csv_file in csv_files:
        print(f"  {csv_file}")
    
    if args.dry_run:
        print("\n干运行模式 - 不会实际修改文件")
        return
    
    print(f"\n开始处理，wav基础路径: {args.wav_base_path}")
    print(f"备份设置: {'禁用' if args.no_backup else '启用'}")
    print("-" * 50)
    
    # 处理每个CSV文件
    processed_count = 0
    for csv_file in csv_files:
        process_csv_file(csv_file, args.wav_base_path, backup=not args.no_backup)
        processed_count += 1
        print()
    
    print("-" * 50)
    print(f"处理完成! 总共处理了 {processed_count} 个文件")

if __name__ == "__main__":
    main()
