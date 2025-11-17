#!/usr/bin/env python3
"""
简化版本：将当前目录和子目录中所有CSV文件的特征列替换为wav_path列
"""

import pandas as pd
import os
import glob

def create_wav_path_from_prosody(prosody_path):
    """
    根据prosody路径创建wav路径
    例如：./data/gita/speech_features/disvoice/prosody/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz
    转换为：/data/storage1t/data_nofolder/gita/norm_audios/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.wav
    """
    if pd.isna(prosody_path) or prosody_path == '':
        return ''
    
    # 分割路径并获取文件名
    filename = prosody_path.split('/')[-1].replace('.npz', '.wav')
    
    # 构建新路径
    wav_path = '/data/storage1t/data_nofolder/gita/norm_audios/' + filename
    
    return wav_path

def process_csv_files():
    """处理所有CSV文件"""
    
    # 查找当前目录及子目录中的所有CSV文件
    csv_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for csv_file in csv_files:
        print(f"  {csv_file}")
    
    # 要删除的特征列
    feature_columns = ['prosody', 'wav2vec', 'phonation', 'articulation', 'glottal']
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        try:
            print(f"\n处理文件: {csv_file}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否存在prosody列
            if 'prosody' not in df.columns:
                print(f"  跳过: 没有找到prosody列")
                continue
            
            # 创建备份
            backup_file = csv_file + '.backup'
            if not os.path.exists(backup_file):
                df.to_csv(backup_file, index=False)
                print(f"  创建备份: {backup_file}")
            
            # 创建wav_path列
            df['wav_path'] = df['prosody'].apply(create_wav_path_from_prosody)
            
            # 删除特征列（如果存在）
            columns_to_drop = [col for col in feature_columns if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                print(f"  删除列: {columns_to_drop}")
            
            # 保存文件
            df.to_csv(csv_file, index=False)
            print(f"  成功处理 {len(df)} 行数据")
            
        except Exception as e:
            print(f"  错误: {str(e)}")

if __name__ == "__main__":
    print("开始处理CSV文件...")
    print("=" * 50)
    process_csv_files()
    print("=" * 50)
    print("处理完成!")
