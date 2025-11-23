import requests
import json
import time
import pandas as pd
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = "cache"
RAW_DATA_DIR = "cache/raw_api_responses"  # Store raw API responses
CACHE_ENABLED = True
CHECKPOINT_INTERVAL = 50
CACHE_VERSION = "v2"  # Increment when extraction logic changes

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

print("""
╔══════════════════════════════════════════════════════════╗
║       FACEIT PLAYER STATISTICS EXTRACTOR (FIXED v2)     ║
║                                                          ║
║  Features:                                               ║
║  • Aggregates detailed stats from map segments          ║
║  • Uses lifetime stats where available                  ║
║  • Saves progress every 50 players                      ║
║  • Resumes from last checkpoint on error                ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# RAW DATA CACHE FUNCTIONS
# ============================================================

def get_raw_data_path(player_id: str) -> str:
    """Get path for raw API response"""
    return os.path.join(RAW_DATA_DIR, f"{player_id}_raw.json")

def save_raw_data(player_id: str, profile: Dict, stats: Dict):
    """Save raw API responses to disk"""
    raw_data_path = get_raw_data_path(player_id)
    try:
        raw_data = {
            "player_id": player_id,
            "timestamp": time.time(),
            "profile": profile,
            "stats": stats
        }
        with open(raw_data_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to save raw data for {player_id}: {e}")

def load_raw_data(player_id: str) -> Optional[Dict]:
    """Load raw API responses from disk"""
    raw_data_path = get_raw_data_path(player_id)
    if os.path.exists(raw_data_path):
        try:
            with open(raw_data_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

# ============================================================
# CACHE FUNCTIONS
# ============================================================

def get_cache_path(player_id: str) -> str:
    """Get cache file path for a player with version"""
    return os.path.join(CACHE_DIR, f"{player_id}_{CACHE_VERSION}.json")

def load_from_cache(player_id: str) -> Optional[Dict]:
    """Load player data from cache"""
    cache_path = get_cache_path(player_id)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_to_cache(player_id: str, data: Dict):
    """Save player data to cache"""
    cache_path = get_cache_path(player_id)
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️  Failed to cache {player_id}: {e}")

def get_checkpoint_path(player_class: str) -> str:
    """Get checkpoint file path"""
    return os.path.join(CACHE_DIR, f"checkpoint_{player_class}_{CACHE_VERSION}.json")

def save_checkpoint(player_class: str, features_list: List[Dict], processed_ids: List[str]):
    """Save checkpoint with current progress"""
    checkpoint_path = get_checkpoint_path(player_class)
    checkpoint_data = {
        "features": features_list,
        "processed_ids": processed_ids,
        "timestamp": time.time()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"💾 Checkpoint saved: {len(features_list)} players")

def load_checkpoint(player_class: str) -> Optional[Dict]:
    """Load checkpoint if exists"""
    checkpoint_path = get_checkpoint_path(player_class)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


# ============================================================
# FETCH PLAYER DATA
# ============================================================

def get_player_profile(player_id: str) -> Optional[Dict]:
    """Fetch player basic profile information"""
    url = f"{config.BASE_URL}/players/{player_id}"
    
    try:
        response = requests.get(url, headers=config.HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def get_player_stats(player_id: str) -> Optional[Dict]:
    """Fetch player CS2 statistics"""
    url = f"{config.BASE_URL}/players/{player_id}/stats/{config.GAME_ID}"
    
    try:
        response = requests.get(url, headers=config.HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


# ============================================================
# EXTRACT FEATURES - CORRECTED VERSION
# ============================================================

def extract_features(player_id: str, player_class: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Extract features from player data
    Combines lifetime stats + aggregated segment stats
    """
    
    # Try processed cache first
    if use_cache and CACHE_ENABLED:
        cached_data = load_from_cache(player_id)
        if cached_data:
            cached_data["player_class"] = player_class
            return cached_data
    
    # Try to load from raw data cache (avoid API calls)
    raw_data = load_raw_data(player_id)
    
    if raw_data:
        print(f"📦 Using cached raw data for {player_id}")
        profile = raw_data.get("profile")
        stats = raw_data.get("stats")
    else:
        # Fetch fresh data from API
        profile = get_player_profile(player_id)
        if not profile:
            return None
        
        stats = get_player_stats(player_id)
        if not stats:
            return None
        
        # Save raw API responses for future use
        save_raw_data(player_id, profile, stats)
        # print(f"💾 Saved raw data for {player_id}")
    
    try:
        # Basic info from profile
        games = profile.get("games", {}).get(config.GAME_ID, {})
        skill_level = games.get("skill_level", 0)
        faceit_elo = games.get("faceit_elo", 0)
        
        # Get lifetime stats
        lifetime = stats.get("lifetime", {})
        if not lifetime:
            print(f"⚠️  No lifetime stats for {player_id}")
            return None
        
        # Get segments (map-specific stats)
        segments = stats.get("segments", [])
        
        # Helper function to safely get stat value from lifetime
        def get_lifetime_stat(stat_name: str, default=0):
            stat_value = lifetime.get(stat_name, default)
            if isinstance(stat_value, str):
                try:
                    if stat_value.endswith('%'):
                        stat_value = stat_value[:-1]
                    return float(stat_value)
                except:
                    return default
            return float(stat_value) if stat_value is not None else default
        
        # Get matches played
        matches = get_lifetime_stat("Matches")
        if matches == 0:
            print(f"⚠️  No matches for {player_id}")
            return None
        
        # ============================================================
        # AGGREGATE STATS FROM SEGMENTS (per-map stats)
        # ============================================================
        
        total_kills = 0
        total_deaths = 0
        total_assists = 0
        total_headshots = 0
        total_triple_kills = 0
        total_quadro_kills = 0
        total_penta_kills = 0
        total_mvps = 0
        
        for segment in segments:
            seg_stats = segment.get("stats", {})
            
            # Helper to get segment stat
            def get_seg_stat(name, default=0):
                val = seg_stats.get(name, default)
                try:
                    return float(val) if val else default
                except:
                    return default
            
            # Aggregate totals
            total_kills += get_seg_stat("Kills")
            total_deaths += get_seg_stat("Deaths")
            total_assists += get_seg_stat("Assists")
            total_headshots += get_seg_stat("Headshots")
            total_triple_kills += get_seg_stat("Triple Kills")
            total_quadro_kills += get_seg_stat("Quadro Kills")
            total_penta_kills += get_seg_stat("Penta Kills")
            total_mvps += get_seg_stat("MVPs")
        
        # ============================================================
        # BUILD FEATURE DICTIONARY
        # ============================================================
        
        # Calculate averages
        avg_kills = total_kills / matches if matches > 0 else 0
        avg_deaths = total_deaths / matches if matches > 0 else 0
        avg_assists = total_assists / matches if matches > 0 else 0
        avg_mvps = total_mvps / matches if matches > 0 else 0
        
        # Recent form from Recent Results
        recent_results = lifetime.get("Recent Results", [])
        if len(recent_results) > 0:
            recent_wins = sum(1 for r in recent_results if int(r) == 1)
            recent_win_rate = (recent_wins / len(recent_results)) * 100
        else:
            recent_win_rate = get_lifetime_stat("Win Rate %")
        
        features = {
            "player_id": player_id,
            "player_class": player_class,
            
            # Profile stats
            "skill_level": skill_level,
            "faceit_elo": faceit_elo,
            
            # Lifetime stats (directly from API)
            "matches": matches,
            "wins": get_lifetime_stat("Wins"),
            "win_rate": get_lifetime_stat("Win Rate %"),
            "kd_ratio": get_lifetime_stat("Average K/D Ratio"),
            "avg_headshots": get_lifetime_stat("Average Headshots %"),
            "adr": get_lifetime_stat("ADR"),
            
            # Calculated from segments
            "avg_kills": avg_kills,
            "avg_deaths": avg_deaths,
            "avg_assists": avg_assists,
            "avg_mvps": avg_mvps,
            
            # Totals from segments
            "total_kills": total_kills,
            "total_deaths": total_deaths,
            "total_assists": total_assists,
            "total_headshots": total_headshots,
            "triple_kills": total_triple_kills,
            "quadro_kills": total_quadro_kills,
            "penta_kills": total_penta_kills,
            
            # Streaks from lifetime
            "current_win_streak": get_lifetime_stat("Current Win Streak"),
            "longest_win_streak": get_lifetime_stat("Longest Win Streak"),
            
            # Per-match averages
            "kills_per_match": total_kills / matches,
            "deaths_per_match": total_deaths / matches,
            "assists_per_match": total_assists / matches,
            
            # Recent form
            "recent_win_rate": recent_win_rate,
            
            # Additional useful stats from lifetime
            "entry_success_rate": get_lifetime_stat("Entry Success Rate"),
            "clutch_1v1_win_rate": get_lifetime_stat("1v1 Win Rate"),
            "clutch_1v2_win_rate": get_lifetime_stat("1v2 Win Rate"),
            "flash_success_rate": get_lifetime_stat("Flash Success Rate"),
            "utility_damage_per_round": get_lifetime_stat("Utility Damage per Round"),
        }
        
        # Cache the features (without player_class)
        if CACHE_ENABLED:
            cache_data = {k: v for k, v in features.items() if k != "player_class"}
            save_to_cache(player_id, cache_data)
        
        return features
    
    except Exception as e:
        print(f"❌ Error extracting features for {player_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# PROCESS PLAYER CLASS WITH CHECKPOINTS
# ============================================================

def process_player_class(player_ids: List[str], player_class: str) -> List[Dict]:
    """Process all players in a class with checkpoint support"""
    
    # Try to load checkpoint
    checkpoint = load_checkpoint(player_class)
    
    if checkpoint:
        print(f"\n🔄 Found checkpoint for {player_class} class")
        print(f"   Resuming from {len(checkpoint['features'])} players")
        features_list = checkpoint["features"]
        processed_ids = set(checkpoint["processed_ids"])
        
        # Filter out already processed IDs
        remaining_ids = [pid for pid in player_ids if pid not in processed_ids]
        print(f"   {len(remaining_ids)} players remaining")
    else:
        features_list = []
        processed_ids = set()
        remaining_ids = player_ids
    
    failed_count = 0
    
    print(f"\n📊 Processing {len(remaining_ids)} {player_class} players...")
    
    for idx, player_id in enumerate(tqdm(remaining_ids, desc=f"{player_class} players")):
        features = extract_features(player_id, player_class, use_cache=True)
        
        if features and features["matches"] > 0:
            features_list.append(features)
            processed_ids.add(player_id)
        else:
            failed_count += 1
        
        # Save checkpoint every N players
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(player_class, features_list, list(processed_ids))
        
        # Rate limiting
        time.sleep(0.15)
    
    # Final checkpoint
    save_checkpoint(player_class, features_list, list(processed_ids))
    
    print(f"✅ Successfully processed: {len(features_list)}/{len(player_ids)}")
    print(f"❌ Failed or empty: {failed_count}")
    
    return features_list


# ============================================================
# MAIN PIPELINE
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
        print("⚠️  config.PRO_IDS_JSON not found")
        pro_ids = []
    
    try:
        with open(config.HIGH_LEVEL_IDS_JSON, "r") as f:
            high_ids = json.load(f)
        print(f"✅ Loaded {len(high_ids)} HIGH-LEVEL player IDs")
    except FileNotFoundError:
        print("⚠️  config.HIGH_LEVEL_IDS_JSON not found")
        high_ids = []
    
    try:
        with open(config.NORMAL_IDS_JSON, "r") as f:
            normal_ids = json.load(f)
        print(f"✅ Loaded {len(normal_ids)} NORMAL player IDs")
    except FileNotFoundError:
        print("⚠️  config.NORMAL_IDS_JSON not found")
        normal_ids = []
    
    # Check cache status
    cache_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith(f'_{CACHE_VERSION}.json')])
    raw_files = len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('_raw.json')])
    print(f"\n💾 Processed cache: {cache_files} players (version {CACHE_VERSION})")
    print(f"📦 Raw API cache: {raw_files} players")
    
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
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"\n📋 Class distribution:")
    print(df["player_class"].value_counts())
    
    # Display sample statistics per class
    print("\n" + "="*60)
    print("SAMPLE STATISTICS BY CLASS")
    print("="*60)
    for player_class in df["player_class"].unique():
        class_df = df[df["player_class"] == player_class]
        print(f"\n{player_class}:")
        print(f"  Players: {len(class_df)}")
        print(f"  Avg ELO: {class_df['faceit_elo'].mean():.0f}")
        print(f"  Avg K/D: {class_df['kd_ratio'].mean():.2f}")
        print(f"  Avg Kills/Match: {class_df['avg_kills'].mean():.2f}")
        print(f"  Avg Deaths/Match: {class_df['avg_deaths'].mean():.2f}")
        print(f"  Avg Win Rate: {class_df['win_rate'].mean():.1f}%")
        print(f"  Avg Matches: {class_df['matches'].mean():.0f}")
    
    # Check for zeros
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    zero_cols = df.columns[(df == 0).sum() > len(df) * 0.5]  # Columns with >50% zeros
    if len(zero_cols) > 0:
        print(f"⚠️  Columns with many zeros: {list(zero_cols)}")
    else:
        print("✅ No suspicious zero columns detected")
    
    # Save dataset
    df.to_csv(config.DATASET_CSV, index=False)
    print(f"\n✅ Dataset saved to: {config.DATASET_CSV}")
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE DATA (First 3 rows)")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(3))
    
    # Clean up checkpoints (optional)
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    response = input("Delete checkpoints? (they're no longer needed) [y/N]: ")
    if response.lower() == 'y':
        for player_class in ['PRO', 'HIGH_LEVEL', 'NORMAL']:
            checkpoint_path = get_checkpoint_path(player_class)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"✅ Deleted checkpoint for {player_class}")
    
    print("\n✅ Feature extraction complete! Ready for preprocessing.")
    print(f"\n💡 Cache info:")
    print(f"   • Processed features: {cache_files} files in cache/")
    print(f"   • Raw API responses: {raw_files} files in cache/raw_api_responses/")
    print(f"\n💡 If extraction logic changes:")
    print(f"   1. Increment CACHE_VERSION in the script")
    print(f"   2. Re-run - it will use raw API cache (no new requests!)")
    print(f"   3. Only missing players will trigger new API calls")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Your progress is saved in checkpoints.")
        print("   Just run the script again to resume from where you left off.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("   Your progress is saved! Run again to resume.")