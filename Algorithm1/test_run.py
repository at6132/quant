#!/usr/bin/env python3
import yaml
import sys
import traceback
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).parent))

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Config loaded successfully")

# Try imports
try:
    from src.ingestion.load_multitf import load_all_frames, validate_frames
    print("Import 1: OK - load_multitf")
except Exception as e:
    print(f"Import 1 FAILED: {e}")
    traceback.print_exc()

try:
    from src.feature_engineering.engineer_features import engineer_features
    print("Import 2: OK - engineer_features")
except Exception as e:
    print(f"Import 2 FAILED: {e}")
    traceback.print_exc()

try:
    from src.labeling.generate_labels import generate_labels
    print("Import 3: OK - generate_labels")
except Exception as e:
    print(f"Import 3 FAILED: {e}")
    traceback.print_exc()

try:
    from src.models.train_models import train_models
    print("Import 4: OK - train_models")
except Exception as e:
    print(f"Import 4 FAILED: {e}")
    traceback.print_exc()

# Try loading data
try:
    print("\nStep 1: Loading data...")
    raw_data_dict = load_all_frames(config)
    validate_frames(raw_data_dict)
    raw_data = raw_data_dict[config['timeframes'][0]]
    print(f"Data loaded: {raw_data.shape}")
except Exception as e:
    print(f"Data loading FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try feature engineering
try:
    print("\nStep 2: Engineering features...")
    features = engineer_features(raw_data, config)
    print(f"Features engineered: {features.shape}")
except Exception as e:
    print(f"Feature engineering FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try label generation
try:
    print("\nStep 3: Generating labels...")
    labeled_data = generate_labels(features, config)
    print(f"Labels generated: {labeled_data.shape}")
    print(f"Label distribution: {labeled_data['label'].value_counts()}")
except Exception as e:
    print(f"Label generation FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All steps completed successfully!")