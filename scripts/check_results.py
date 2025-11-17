#!/usr/bin/env python3
"""
总结脚本：显示CSV文件处理的统计信息
"""

import pandas as pd
import os

def get_csv_statistics():
    """获取CSV文件处理后的统计信息"""
    
    csv_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.csv') and not file.endswith('.backup'):
                csv_files.append(os.path.join(root, file))
    
    print(f"处理后的CSV文件统计")
    print("=" * 50)
    print(f"总文件数: {len(csv_files)}")
    
    total_rows = 0
    sample_files = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            total_rows += len(df)
            
            # 取前几个文件作为样本
            if i < 3:
                sample_files.append((csv_file, df))
        except Exception as e:
            print(f"读取文件出错 {csv_file}: {e}")
    
    print(f"总数据行数: {total_rows:,}")
    
    print("\n样本文件结构:")
    print("-" * 50)
    
    for file_path, df in sample_files:
        print(f"\n文件: {file_path}")
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        
        if 'wav_path' in df.columns:
            print(f"wav_path样本: {df['wav_path'].iloc[0]}")
        
        print("-" * 30)

if __name__ == "__main__":
    get_csv_statistics()
