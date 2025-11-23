import os
import config

def verify():
    print("Testing data paths...")
    
    # 1. Check Data Directory
    if os.path.exists(config.DATA_DIR):
        print(f"✅ Data directory exists: {config.DATA_DIR}")
    else:
        print(f"❌ Data directory missing: {config.DATA_DIR}")

    # 2. Check Data Files (if they exist)
    files_to_check = [
        ("PRO_DATA_CSV", config.PRO_DATA_CSV),
        ("PRO_IDS_JSON", config.PRO_IDS_JSON),
        ("HIGH_LEVEL_IDS_JSON", config.HIGH_LEVEL_IDS_JSON),
        ("NORMAL_IDS_JSON", config.NORMAL_IDS_JSON),
        ("DATASET_CSV", config.DATASET_CSV)
    ]

    for name, path in files_to_check:
        if os.path.exists(path):
            print(f"✅ {name} found at: {path}")
        else:
            print(f"⚠️  {name} not found (might not be generated yet)")

    print("\nTesting imports...")
    try:
        import pro_players
        print("✅ pro_players imported successfully")
    except Exception as e:
        print(f"❌ pro_players import failed: {e}")

    try:
        import data_collection
        print("✅ data_collection imported successfully")
    except Exception as e:
        print(f"❌ data_collection import failed: {e}")

    try:
        import feature_extraction
        print("✅ feature_extraction imported successfully")
    except Exception as e:
        print(f"❌ feature_extraction import failed: {e}")

if __name__ == "__main__":
    verify()
