import requests
import json
import time
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import config

# ============================================================
# CONFIGURATION
# ============================================================

# Configuration is now handled in config.py


print("""
╔══════════════════════════════════════════════════════════╗
║          FACEIT PLAYER STATISTICS EXTRACTOR             ║
║                                                          ║
║  Extracting features for ML classification:             ║
║  • Player profile data                                   ║
║  • Overall statistics (K/D, Win Rate, HS%)              ║
║  • Recent performance metrics                            ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# 1. FETCH PLAYER PROFILE
# ============================================================

def get_player_profile(player_id: str) -> Optional[Dict]:
    """Fetch player basic profile information"""
    url = f"{config.BASE_URL}/players/{player_id}"
    
    try:
        response = requests.get(url, headers=config.HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            print(f"⚠️  Error {response.status_code} for player {player_id}")
            return None
    except Exception as e:
        print(f"❌ Exception fetching profile {player_id}: {e}")
        return None


# ============================================================
# 2. FETCH PLAYER STATISTICS
# ============================================================

def get_player_stats(player_id: str) -> Optional[Dict]:
    """Fetch player CS2 statistics"""
    url = f"{config.BASE_URL}/players/{player_id}/stats/{config.GAME_ID}"
    
    try:
        response = requests.get(url, headers=config.HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            return None
    except Exception as e:
        return None


# ============================================================
# 3. EXTRACT FEATURES FROM PLAYER DATA
# ============================================================

def extract_features(player_id: str, player_class: str) -> Optional[Dict]:
    """
    Extract all relevant features for a player
    
    Features:
    - ELO, Skill Level
    - K/D Ratio, Win Rate, Matches
    - Average Kills, Deaths, Assists per match
    - Headshot %, K/R Ratio
    - Triple/Quadro/Penta kills
    - Recent form (last 20 matches)
    """
    
    # Fetch profile
    profile = get_player_profile(player_id)
    if not profile:
        return None
    
    # Fetch stats
    stats = get_player_stats(player_id)
    if not stats:
        return None
    
    try:
        # Basic info
        games = profile.get("games", {}).get(config.GAME_ID, {})
        skill_level = games.get("skill_level", 0)
        faceit_elo = games.get("faceit_elo", 0)
        
        # Lifetime stats
        lifetime = stats.get("lifetime", {})
        
        # Segments (overall stats)
        segments = stats.get("segments", [])
        overall_stats = {}
        
        for segment in segments:
            if segment.get("type") == "Overall":
                overall_stats = segment.get("stats", {})
                break
        
        # Helper function to safely get stat value
        def get_stat(stat_name: str, default=0):
            stat_obj = overall_stats.get(stat_name, {})
            if isinstance(stat_obj, dict):
                return float(stat_obj.get("value", default))
            return default
        
        # Extract features
        features = {
            # Player ID and class
            "player_id": player_id,
            "player_class": player_class,
            
            # Basic info
            "skill_level": skill_level,
            "faceit_elo": faceit_elo,
            
            # Match statistics
            "matches": get_stat("Matches"),
            "wins": get_stat("Wins"),
            "win_rate": get_stat("Win Rate %"),
            
            # Performance metrics
            "kd_ratio": get_stat("Average K/D Ratio"),
            "kr_ratio": get_stat("Average K/R Ratio"),
            "avg_kills": get_stat("Average Kills"),
            "avg_deaths": get_stat("Average Deaths"),
            "avg_assists": get_stat("Average Assists"),
            "avg_headshots": get_stat("Average Headshots %"),
            "avg_mvps": get_stat("Average MVPs"),
            
            # Kill statistics
            "total_kills": get_stat("Total Kills"),
            "total_deaths": get_stat("Total Deaths"),
            "total_assists": get_stat("Assists"),
            "total_headshots": get_stat("Total Headshots %"),
            
            # Special kills
            "triple_kills": get_stat("Triple Kills"),
            "quadro_kills": get_stat("Quadro Kills"),
            "penta_kills": get_stat("Penta Kills"),
            
            # Recent matches (from lifetime)
            "recent_results": lifetime.get("Recent Results", []),
            "current_win_streak": get_stat("Current Win Streak"),
            "longest_win_streak": get_stat("Longest Win Streak"),
        }
        
        # Calculate derived features
        if features["matches"] > 0:
            features["kills_per_match"] = features["total_kills"] / features["matches"]
            features["deaths_per_match"] = features["total_deaths"] / features["matches"]
            features["assists_per_match"] = features["total_assists"] / features["matches"]
        else:
            features["kills_per_match"] = 0
            features["deaths_per_match"] = 0
            features["assists_per_match"] = 0
        
        # Recent form (win rate in last 20 matches)
        recent_results = features["recent_results"]
        if len(recent_results) > 0:
            recent_wins = sum(1 for r in recent_results if r == "1")
            features["recent_win_rate"] = (recent_wins / len(recent_results)) * 100
        else:
            features["recent_win_rate"] = features["win_rate"]
        
        return features
    
    except Exception as e:
        print(f"❌ Error extracting features for {player_id}: {e}")
        return None


# ============================================================
# 4. PROCESS ALL PLAYERS
# ============================================================

def process_player_class(player_ids: List[str], player_class: str) -> List[Dict]:
    """Process all players in a class"""
    features_list = []
    failed_count = 0
    
    print(f"\n📊 Processing {len(player_ids)} {player_class} players...")
    
    for player_id in tqdm(player_ids, desc=f"{player_class} players"):
        features = extract_features(player_id, player_class)
        
        if features:
            features_list.append(features)
        else:
            failed_count += 1
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"✅ Successfully processed: {len(features_list)}/{len(player_ids)}")
    print(f"❌ Failed: {failed_count}")
    
    return features_list


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "="*60)
    print("LOADING PLAYER IDs")
    print("="*60)
    
    # Load player IDs
    try:
        with open(config.PRO_IDS_JSON, "r") as f:
            pro_ids = json.load(f)
        print(f"✅ Loaded {len(pro_ids)} PRO player IDs")
    except FileNotFoundError:
        print(f"⚠️  {config.PRO_IDS_JSON} not found")
        pro_ids = []
    
    try:
        with open(config.HIGH_LEVEL_IDS_JSON, "r") as f:
            high_ids = json.load(f)
        print(f"✅ Loaded {len(high_ids)} HIGH-LEVEL player IDs")
    except FileNotFoundError:
        print(f"⚠️  {config.HIGH_LEVEL_IDS_JSON} not found")
        high_ids = []
    
    try:
        with open(config.NORMAL_IDS_JSON, "r") as f:
            normal_ids = json.load(f)
        print(f"✅ Loaded {len(normal_ids)} NORMAL player IDs")
    except FileNotFoundError:
        print(f"⚠️  {config.NORMAL_IDS_JSON} not found")
        normal_ids = []
    
    # Process each class
    all_features = []
    
    # PRO
    if pro_ids:
        pro_features = process_player_class(pro_ids, "PRO")
        all_features.extend(pro_features)
    
    # HIGH-LEVEL
    if high_ids:
        high_features = process_player_class(high_ids, "HIGH_LEVEL")
        all_features.extend(high_features)
    
    # NORMAL
    if normal_ids:
        normal_features = process_player_class(normal_ids, "NORMAL")
        all_features.extend(normal_features)
    
    # Create DataFrame
    print("\n" + "="*60)
    print("CREATING DATASET")
    print("="*60)
    
    df = pd.DataFrame(all_features)
    
    # Remove recent_results column (not needed for ML)
    if "recent_results" in df.columns:
        df = df.drop(columns=["recent_results"])
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"\n📋 Class distribution:")
    print(df["player_class"].value_counts())
    
    # Save dataset
    # Save dataset
    df.to_csv(config.DATASET_CSV, index=False)
    print(f"\n✅ Dataset saved to: {config.DATASET_CSV}")
    
    # Display sample statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(df.describe())
    
    print("\n" + "="*60)
    print("MISSING VALUES")
    print("="*60)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ No missing values!")
    
    print("\n✅ Feature extraction complete! Ready for preprocessing.")


if __name__ == "__main__":
    main()


# ╔══════════════════════════════════════════════════════════╗
# ║          FACEIT PLAYER STATISTICS EXTRACTOR             ║
# ║                                                          ║
# ║  Extracting features for ML classification:             ║
# ║  • Player profile data                                   ║
# ║  • Overall statistics (K/D, Win Rate, HS%)              ║
# ║  • Recent performance metrics                            ║
# ╚══════════════════════════════════════════════════════════╝


# ============================================================
# LOADING PLAYER IDs
# ============================================================
# ✅ Loaded 577 PRO player IDs
# ✅ Loaded 796 HIGH-LEVEL player IDs
# ✅ Loaded 747 NORMAL player IDs

# 📊 Processing 577 PRO players...
# PRO players: 100%|██████████████████████████████████████████████████████████████████████████████| 577/577 [16:13<00:00,  1.69s/it]
# ✅ Successfully processed: 577/577
# ❌ Failed: 0

# 📊 Processing 796 HIGH_LEVEL players...
# HIGH_LEVEL players: 100%|███████████████████████████████████████████████████████████████████████| 796/796 [22:45<00:00,  1.72s/it]
# ✅ Successfully processed: 795/796
# ❌ Failed: 1

# 📊 Processing 747 NORMAL players...
# NORMAL players: 100%|███████████████████████████████████████████████████████████████████████████| 747/747 [20:58<00:00,  1.68s/it]
# ✅ Successfully processed: 745/747
# ❌ Failed: 2

# ============================================================
# CREATING DATASET
# ============================================================

# 📊 Dataset shape: (2117, 27)

# 📋 Class distribution:
# player_class
# HIGH_LEVEL    795
# NORMAL        745
# PRO           577
# Name: count, dtype: int64

# ✅ Dataset saved to: /home/ansinitro/faceit-classifier/data/faceit_players_dataset.csv

# ============================================================
# DATASET STATISTICS
# ============================================================
#        skill_level   faceit_elo  matches    wins  ...  kills_per_match  deaths_per_match  assists_per_match  recent_win_rate
# count  2117.000000  2117.000000   2117.0  2117.0  ...           2117.0            2117.0             2117.0      2117.000000
# mean      8.590931  2649.270666      0.0     0.0  ...              0.0               0.0                0.0        55.692017
# std       1.910787  1013.469789      0.0     0.0  ...              0.0               0.0                0.0        23.380545
# min       6.000000  1236.000000      0.0     0.0  ...              0.0               0.0                0.0         0.000000
# 25%       6.000000  1333.000000      0.0     0.0  ...              0.0               0.0                0.0        40.000000
# 50%      10.000000  3098.000000      0.0     0.0  ...              0.0               0.0                0.0        60.000000
# 75%      10.000000  3553.000000      0.0     0.0  ...              0.0               0.0                0.0        80.000000
# max      10.000000  4161.000000      0.0     0.0  ...              0.0               0.0                0.0       100.000000

# [8 rows x 25 columns]

# ============================================================
# MISSING VALUES
# ============================================================
# ✅ No missing values!

# ✅ Feature extraction complete! Ready for preprocessing.╔══════════════════════════════════════════════════════════╗
# ║          FACEIT PLAYER STATISTICS EXTRACTOR             ║
# ║                                                          ║
# ║  Extracting features for ML classification:             ║
# ║  • Player profile data                                   ║
# ║  • Overall statistics (K/D, Win Rate, HS%)              ║
# ║  • Recent performance metrics                            ║
# ╚══════════════════════════════════════════════════════════╝


# ============================================================
# LOADING PLAYER IDs
# ============================================================
# ✅ Loaded 577 PRO player IDs
# ✅ Loaded 796 HIGH-LEVEL player IDs
# ✅ Loaded 747 NORMAL player IDs

# 📊 Processing 577 PRO players...
# PRO players: 100%|██████████████████████████████████████████████████████████████████████████████| 577/577 [16:13<00:00,  1.69s/it]
# ✅ Successfully processed: 577/577
# ❌ Failed: 0

# 📊 Processing 796 HIGH_LEVEL players...
# HIGH_LEVEL players: 100%|███████████████████████████████████████████████████████████████████████| 796/796 [22:45<00:00,  1.72s/it]
# ✅ Successfully processed: 795/796
# ❌ Failed: 1

# 📊 Processing 747 NORMAL players...
# NORMAL players: 100%|███████████████████████████████████████████████████████████████████████████| 747/747 [20:58<00:00,  1.68s/it]
# ✅ Successfully processed: 745/747
# ❌ Failed: 2

# ============================================================
# CREATING DATASET
# ============================================================

# 📊 Dataset shape: (2117, 27)

# 📋 Class distribution:
# player_class
# HIGH_LEVEL    795
# NORMAL        745
# PRO           577
# Name: count, dtype: int64

# ✅ Dataset saved to: /home/ansinitro/faceit-classifier/data/faceit_players_dataset.csv

# ============================================================
# DATASET STATISTICS
# ============================================================
#        skill_level   faceit_elo  matches    wins  ...  kills_per_match  deaths_per_match  assists_per_match  recent_win_rate
# count  2117.000000  2117.000000   2117.0  2117.0  ...           2117.0            2117.0             2117.0      2117.000000
# mean      8.590931  2649.270666      0.0     0.0  ...              0.0               0.0                0.0        55.692017
# std       1.910787  1013.469789      0.0     0.0  ...              0.0               0.0                0.0        23.380545
# min       6.000000  1236.000000      0.0     0.0  ...              0.0               0.0                0.0         0.000000
# 25%       6.000000  1333.000000      0.0     0.0  ...              0.0               0.0                0.0        40.000000
# 50%      10.000000  3098.000000      0.0     0.0  ...              0.0               0.0                0.0        60.000000
# 75%      10.000000  3553.000000      0.0     0.0  ...              0.0               0.0                0.0        80.000000
# max      10.000000  4161.000000      0.0     0.0  ...              0.0               0.0                0.0       100.000000

# [8 rows x 25 columns]

# ============================================================
# MISSING VALUES
# ============================================================
# ✅ No missing values!

# ╔══════════════════════════════════════════════════════════╗
# ║          FACEIT PLAYER STATISTICS EXTRACTOR             ║
# ║                                                          ║
# ║  Extracting features for ML classification:             ║
# ║  • Player profile data                                   ║
# ║  • Overall statistics (K/D, Win Rate, HS%)              ║
# ║  • Recent performance metrics                            ║
# ╚══════════════════════════════════════════════════════════╝


# ============================================================
# LOADING PLAYER IDs
# ============================================================
# ✅ Loaded 577 PRO player IDs
# ✅ Loaded 796 HIGH-LEVEL player IDs
# ✅ Loaded 747 NORMAL player IDs

# 📊 Processing 577 PRO players...
# PRO players: 100%|██████████████████████████████████████████████████████████████████████████████| 577/577 [16:13<00:00,  1.69s/it]
# ✅ Successfully processed: 577/577
# ❌ Failed: 0

# 📊 Processing 796 HIGH_LEVEL players...
# HIGH_LEVEL players: 100%|███████████████████████████████████████████████████████████████████████| 796/796 [22:45<00:00,  1.72s/it]
# ✅ Successfully processed: 795/796
# ❌ Failed: 1

# 📊 Processing 747 NORMAL players...
# NORMAL players: 100%|███████████████████████████████████████████████████████████████████████████| 747/747 [20:58<00:00,  1.68s/it]
# ✅ Successfully processed: 745/747
# ❌ Failed: 2

# ============================================================
# CREATING DATASET
# ============================================================

# 📊 Dataset shape: (2117, 27)

# 📋 Class distribution:
# player_class
# HIGH_LEVEL    795
# NORMAL        745
# PRO           577
# Name: count, dtype: int64

# ✅ Dataset saved to: /home/ansinitro/faceit-classifier/data/faceit_players_dataset.csv

# ============================================================
# DATASET STATISTICS
# ============================================================
#        skill_level   faceit_elo  matches    wins  ...  kills_per_match  deaths_per_match  assists_per_match  recent_win_rate
# count  2117.000000  2117.000000   2117.0  2117.0  ...           2117.0            2117.0             2117.0      2117.000000
# mean      8.590931  2649.270666      0.0     0.0  ...              0.0               0.0                0.0        55.692017
# std       1.910787  1013.469789      0.0     0.0  ...              0.0               0.0                0.0        23.380545
# min       6.000000  1236.000000      0.0     0.0  ...              0.0               0.0                0.0         0.000000
# 25%       6.000000  1333.000000      0.0     0.0  ...              0.0               0.0                0.0        40.000000
# 50%      10.000000  3098.000000      0.0     0.0  ...              0.0               0.0                0.0        60.000000
# 75%      10.000000  3553.000000      0.0     0.0  ...              0.0               0.0                0.0        80.000000
# max      10.000000  4161.000000      0.0     0.0  ...              0.0               0.0                0.0       100.000000

# [8 rows x 25 columns]

# ============================================================
# MISSING VALUES
# ============================================================
# ✅ No missing values!

# ✅ Feature extraction complete! Ready for preprocessing.✅ Feature extraction complete! Ready for preprocessing.