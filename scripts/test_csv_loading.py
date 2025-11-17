#!/usr/bin/env python
"""
Test script to verify CSV format data loading
"""
import sys
import os
import argparse

sys.path.append('/home/yzhong/gits/BDHPD_multi')

# Add the config argument manually for testing
sys.argv = ['test_csv_loading.py', '--config', 'configs/config.yaml']

from yaml_config_manager import load_config
from utils import get_dataset

# Load config
config = load_config()

print(f"Active dataset: {config.active_dataset}")
print(f"PC-GITA active: {config.pc_gita.active}")
print(f"EWADB active: {config.ewadb.active}")

if config.pc_gita.active:
    print("\n=== PC-GITA Configuration ===")
    print(f"Metadata type: {config.pc_gita.metadata_type}")
    print(f"Label key: {config.pc_gita.label_key}")
    print(f"Audio path key: {config.pc_gita.audio_path_key}")
    print(f"Label2ID mapping: {config.pc_gita.label2id}")
    print(f"Train metadata path: {config.pc_gita.train_metadata_path}")
    print(f"Test metadata path: {config.pc_gita.test_metadata_path}")
    print(f"Validation metadata path: {config.pc_gita.validation_metadata_path}")
    
    print("\n=== Testing Dataset Loading ===")
    try:
        # Try loading a small sample of the training data
        train_dataset = get_dataset(config, "train", "pc_gita", domain_id=1)
        print(f"✅ Train dataset loaded successfully!")
        print(f"   Dataset length: {len(train_dataset)}")
        
        # Try loading one sample
        sample = train_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Sample label: {sample['labels']}")
        print(f"   Audio input shape: {sample['input_values'].shape}")
        print(f"   Sample type: {sample['sample_type']}")
        
    except Exception as e:
        print(f"❌ Error loading train dataset: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        # Try loading test data
        test_dataset = get_dataset(config, "test", "pc_gita", domain_id=1)
        print(f"✅ Test dataset loaded successfully!")
        print(f"   Dataset length: {len(test_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        
    try:
        # Try loading validation data
        val_dataset = get_dataset(config, "validation", "pc_gita", domain_id=1)
        print(f"✅ Validation dataset loaded successfully!")
        print(f"   Dataset length: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading validation dataset: {e}")
        
print("\nTest completed!")
