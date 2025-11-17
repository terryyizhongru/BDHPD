#!/usr/bin/env python3
"""
测试脚本：在处理实际文件之前先测试功能
"""

import pandas as pd
import os

def test_conversion():
    """测试转换功能"""
    
    # 创建测试数据
    test_data = {
        'subject_id': ['TEST001'],
        'sample_id': ['TEST_SAMPLE'],
        'task_id': ['TEST'],
        'label': [1],
        'prosody': ['./data/gita/speech_features/disvoice/prosody/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz'],
        'wav2vec': ['./data/gita/speech_features/wav2vec/layer07/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz'],
        'phonation': ['./data/gita/speech_features/disvoice/phonation/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz'],
        'articulation': ['./data/gita/speech_features/disvoice/articulation/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz'],
        'glottal': ['./data/gita/speech_features/disvoice/glottal/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.npz'],
        'UPDRS': [13.0],
        'SEX': ['M']
    }
    
    df = pd.DataFrame(test_data)
    
    print("原始数据:")
    print(df.to_string())
    print(f"\n原始列数: {len(df.columns)}")
    print(f"原始列名: {list(df.columns)}")
    
    # 执行转换
    def create_wav_path_from_prosody(prosody_path):
        if pd.isna(prosody_path) or prosody_path == '':
            return ''
        filename = prosody_path.split('/')[-1].replace('.npz', '.wav')
        wav_path = '/data/storage1t/data_nofolder/gita/norm_audios/' + filename
        return wav_path
    
    # 创建wav_path列
    df['wav_path'] = df['prosody'].apply(create_wav_path_from_prosody)
    
    # 删除特征列
    feature_columns = ['prosody', 'wav2vec', 'phonation', 'articulation', 'glottal']
    df = df.drop(columns=feature_columns)
    
    print("\n" + "="*80)
    print("转换后的数据:")
    print(df.to_string())
    print(f"\n转换后列数: {len(df.columns)}")
    print(f"转换后列名: {list(df.columns)}")
    
    print(f"\nwav_path值: {df['wav_path'].iloc[0]}")
    
    # 验证转换是否正确
    expected_wav_path = '/data/storage1t/data_nofolder/gita/norm_audios/PD_DDK_ANALYSIS_AVPEPUDEA0027_PETAKA.wav'
    actual_wav_path = df['wav_path'].iloc[0]
    
    print(f"\n验证结果:")
    print(f"期望路径: {expected_wav_path}")
    print(f"实际路径: {actual_wav_path}")
    print(f"转换正确: {expected_wav_path == actual_wav_path}")

if __name__ == "__main__":
    print("测试CSV转换功能")
    print("=" * 80)
    test_conversion()
