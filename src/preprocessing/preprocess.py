import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

print("""
╔══════════════════════════════════════════════════════════╗
║              DATA PREPROCESSING PIPELINE                 ║
║                                                          ║
║  Steps:                                                  ║
║  1. Load and clean data                                  ║
║  2. Handle missing values                                ║
║  3. Feature engineering                                  ║
║  4. Encode labels                                        ║
║  5. Scale features                                       ║
║  6. Train/Val/Test split (70/15/15)                      ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(filepath: str = config.DATASET_CSV) -> pd.DataFrame:
    """Load the raw dataset"""
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    print(f"\n📋 Class distribution:")
    print(df["player_class"].value_counts())
    
    return df


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in dataset"""
    print("\n" + "="*60)
    print("STEP 2: HANDLING MISSING VALUES")
    print("="*60)
    
    print(f"\n⚠️  Missing values before cleaning:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ No missing values found!")
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Drop rows with missing critical data
    df = df.dropna(subset=["player_id", "player_class"])
    
    print(f"\n✅ Missing values after cleaning:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ All missing values handled!")
    
    print(f"\n📊 Dataset shape after cleaning: {df.shape}")
    
    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived features"""
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    # Ensure no division by zero
    df["matches"] = df["matches"].replace(0, 1)
    
    # Performance efficiency metrics
    df["kill_efficiency"] = df["total_kills"] / df["matches"]
    df["survival_rate"] = 1 - (df["total_deaths"] / df["matches"])
    df["impact_score"] = (df["avg_kills"] + df["avg_assists"] * 0.5) - df["avg_deaths"]
    
    # Win performance correlation
    df["win_contribution"] = df["win_rate"] * df["kd_ratio"]
    
    # Special kills frequency
    df["special_kills_total"] = df["triple_kills"] + df["quadro_kills"] * 2 + df["penta_kills"] * 3
    df["special_kills_rate"] = df["special_kills_total"] / df["matches"]
    
    # Consistency metric
    df["consistency"] = df["recent_win_rate"] - df["win_rate"]  # positive = improving
    
    # Headshot efficiency
    df["headshot_skill"] = df["avg_headshots"] * df["kd_ratio"]
    
    # Experience level
    df["experience_level"] = np.log1p(df["matches"])
    
    # MVP efficiency
    df["mvp_rate"] = df["avg_mvps"] / df["matches"]
    
    print(f"✅ Created {10} new engineered features")
    print(f"\n📊 New features:")
    print("   • kill_efficiency")
    print("   • survival_rate")
    print("   • impact_score")
    print("   • win_contribution")
    print("   • special_kills_rate")
    print("   • consistency")
    print("   • headshot_skill")
    print("   • experience_level")
    print("   • mvp_rate")
    
    return df


# ============================================================
# 4. SELECT FEATURES FOR TRAINING
# ============================================================

def select_features(df: pd.DataFrame) -> tuple:
    """Select relevant features for model training"""
    print("\n" + "="*60)
    print("STEP 4: FEATURE SELECTION")
    print("="*60)
    
    # Features to use for training
    feature_columns = [
        # Basic info
        "skill_level",
        "faceit_elo",
        "matches",
        
        # Core performance
        "kd_ratio",
        "kr_ratio",
        "win_rate",
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_headshots",
        "avg_mvps",
        
        # Kill stats
        "triple_kills",
        "quadro_kills",
        "penta_kills",
        
        # Streaks
        "current_win_streak",
        "longest_win_streak",
        
        # Derived metrics
        "kills_per_match",
        "deaths_per_match",
        "assists_per_match",
        "recent_win_rate",
        
        # Engineered features
        "kill_efficiency",
        "survival_rate",
        "impact_score",
        "win_contribution",
        "special_kills_rate",
        "consistency",
        "headshot_skill",
        "experience_level",
        "mvp_rate",
    ]
    
    # Filter features that exist in dataframe
    available_features = [f for f in feature_columns if f in df.columns]
    
    print(f"✅ Selected {len(available_features)} features for training")
    
    X = df[available_features]
    y = df["player_class"]
    player_ids = df["player_id"]
    
    return X, y, player_ids, available_features


# ============================================================
# 5. ENCODE LABELS
# ============================================================

def encode_labels(y: pd.Series) -> tuple:
    """Encode class labels to numeric values"""
    print("\n" + "="*60)
    print("STEP 5: ENCODING LABELS")
    print("="*60)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✅ Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"   {label} → {i}")
    
    # Save label encoder
    joblib.dump(label_encoder, config.LABEL_ENCODER)
    print(f"\n💾 Saved label encoder to: {config.LABEL_ENCODER}")
    
    return y_encoded, label_encoder


# ============================================================
# 6. SCALE FEATURES
# ============================================================

def scale_features(X: pd.DataFrame) -> tuple:
    """Standardize features using StandardScaler"""
    print("\n" + "="*60)
    print("STEP 6: SCALING FEATURES")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Scaled {X.shape[1]} features using StandardScaler")
    print(f"\n📊 Feature scaling statistics:")
    print(f"   Mean: {X_scaled.mean():.4f}")
    print(f"   Std:  {X_scaled.std():.4f}")
    
    # Save scaler
    joblib.dump(scaler, config.SCALER)
    print(f"\n💾 Saved scaler to: {config.SCALER}")
    
    return X_scaled, scaler


# ============================================================
# 7. TRAIN/VAL/TEST SPLIT
# ============================================================

def split_data(X, y, player_ids, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets"""
    print("\n" + "="*60)
    print("STEP 7: TRAIN/VAL/TEST SPLIT")
    print("="*60)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, player_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"✅ Data split completed:")
    print(f"   Train set:      {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"   Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_val": ids_val,
        "ids_test": ids_test,
    }


# ============================================================
# 8. SAVE PROCESSED DATA
# ============================================================

def save_processed_data(data_splits, feature_names):
    """Save processed datasets"""
    print("\n" + "="*60)
    print("STEP 8: SAVING PROCESSED DATA")
    print("="*60)
    
    # Save as numpy arrays
    np.save(config.X_TRAIN, data_splits["X_train"])
    np.save(config.X_VAL, data_splits["X_val"])
    np.save(config.X_TEST, data_splits["X_test"])
    np.save(config.Y_TRAIN, data_splits["y_train"])
    np.save(config.Y_VAL, data_splits["y_val"])
    np.save(config.Y_TEST, data_splits["y_test"])
    
    # Save feature names
    with open(config.FEATURE_NAMES_JSON, "w") as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"✅ Saved processed datasets to data/ directory")
    print(f"   • X_train.npy, X_val.npy, X_test.npy")
    print(f"   • y_train.npy, y_val.npy, y_test.npy")
    print(f"   • feature_names.json")


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    # Create directories
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Load data
    df = load_data("./data/faceit_players_dataset.csv")
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Engineer features
    df = engineer_features(df)
    
    # Step 4: Select features
    X, y, player_ids, feature_names = select_features(df)
    
    # Step 5: Encode labels
    y_encoded, label_encoder = encode_labels(y)
    
    # Step 6: Scale features
    X_scaled, scaler = scale_features(X)
    
    # Step 7: Split data
    data_splits = split_data(X_scaled, y_encoded, player_ids)
    
    # Step 8: Save processed data
    save_processed_data(data_splits, feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total samples:     {len(df)}")
    print(f"Total features:    {len(feature_names)}")
    print(f"Classes:           {len(label_encoder.classes_)}")
    print(f"\n📋 Class distribution in training set:")
    train_classes, train_counts = np.unique(data_splits["y_train"], return_counts=True)
    for cls, count in zip(train_classes, train_counts):
        class_name = label_encoder.classes_[cls]
        print(f"   {class_name}: {count} ({count/len(data_splits['y_train'])*100:.1f}%)")
    
    print("\n✅ Preprocessing complete! Ready for model training.")


if __name__ == "__main__":
    main()


# ╔══════════════════════════════════════════════════════════╗
# ║              DATA PREPROCESSING PIPELINE                ║
# ║                                                          ║
# ║  Steps:                                                  ║
# ║  1. Load and clean data                                  ║
# ║  2. Handle missing values                                ║
# ║  3. Feature engineering                                  ║
# ║  4. Encode labels                                        ║
# ║  5. Scale features                                       ║
# ║  6. Train/Val/Test split (70/15/15)                     ║
# ╚══════════════════════════════════════════════════════════╝


# ============================================================
# STEP 1: LOADING DATA
# ============================================================
# ✅ Loaded dataset with shape: (2117, 27)

# 📋 Class distribution:
# player_class
# HIGH_LEVEL    795
# NORMAL        745
# PRO           577
# Name: count, dtype: int64

# ============================================================
# STEP 2: HANDLING MISSING VALUES
# ============================================================

# ⚠️  Missing values before cleaning:
# ✅ No missing values found!

# ✅ Missing values after cleaning:
# ✅ All missing values handled!

# 📊 Dataset shape after cleaning: (2117, 27)

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
# ✅ Created 10 new engineered features

# 📊 New features:
#    • kill_efficiency
#    • survival_rate
#    • impact_score
#    • win_contribution
#    • special_kills_rate
#    • consistency
#    • headshot_skill
#    • experience_level
#    • mvp_rate

# ============================================================
# STEP 4: FEATURE SELECTION
# ============================================================
# ✅ Selected 29 features for training

# ============================================================
# STEP 5: ENCODING LABELS
# ============================================================
# ✅ Label mapping:
#    HIGH_LEVEL → 0
#    NORMAL → 1
#    PRO → 2

# 💾 Saved label encoder to: models/label_encoder.pkl

# ============================================================
# STEP 6: SCALING FEATURES
# ============================================================
# ✅ Scaled 29 features using StandardScaler

# 📊 Feature scaling statistics:
#    Mean: -0.0000
#    Std:  0.3714

# 💾 Saved scaler to: models/scaler.pkl

# ============================================================
# STEP 7: TRAIN/VAL/TEST SPLIT
# ============================================================
# ✅ Data split completed:
#    Train set:      1481 samples (70.0%)
#    Validation set: 318 samples (15.0%)
#    Test set:       318 samples (15.0%)

# ============================================================
# STEP 8: SAVING PROCESSED DATA
# ============================================================
# ✅ Saved processed datasets to data/ directory
#    • X_train.npy, X_val.npy, X_test.npy
#    • y_train.npy, y_val.npy, y_test.npy
#    • feature_names.json

# ============================================================
# PREPROCESSING SUMMARY
# ============================================================
# Total samples:     2117
# Total features:    29
# Classes:           3

# 📋 Class distribution in training set:
#    HIGH_LEVEL: 557 (37.6%)
#    NORMAL: 521 (35.2%)
#    PRO: 403 (27.2%)

# ✅ Preprocessing complete! Ready for model training.