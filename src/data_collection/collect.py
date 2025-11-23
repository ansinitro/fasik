import requests
import json
import csv
import time
import time
from typing import Set, List, Dict
from typing import Set, List, Dict
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# ============================================================
# CONFIGURATION
# ============================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.faceit.com/"
}

# ELO ranges for classification
HIGH_LEVEL_MIN_ELO = 1751  # Level 9-10
NORMAL_MAX_ELO = 1350      # Level 2-6
NORMAL_MIN_ELO = 500       # Level 2+

TARGET_HIGH_LEVEL = 700
TARGET_NORMAL = 700

print("""
╔══════════════════════════════════════════════════════════╗
║     FACEIT CS2 PLAYER CLASSIFICATION DATA COLLECTOR     ║
║                                                          ║
║  Classes:                                                ║
║  • PRO: HLTV professional players                        ║
║  • HIGH-LEVEL: ELO ≥ 1751 (Level 9-10) + FPL           ║
║  • NORMAL: ELO 500-1350 (Level 2-6)                     ║
║                                                          ║
║  Excluded: Level 7-8 (ELO 1351-1750) for clear gaps     ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# 1. LOAD PRO PLAYERS FROM CSV
# ============================================================

# 1. LOAD PRO PLAYERS FROM CSV
# ============================================================

def load_pro_ids(csv_path: str = config.PRO_DATA_CSV) -> Set[str]:
    """Load professional player IDs from HLTV data"""
    pro_ids = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("Faceit Account Link", "").strip()
                if "/users/" in url:
                    pid = url.split("/users/")[-1].split("/")[0]
                    if len(pid) > 10:
                        pro_ids.add(pid)
    except FileNotFoundError:
        print(f"⚠️  WARNING: {csv_path} not found. Starting with 0 pro players.")
        return set()
    
    print(f"✅ Loaded {len(pro_ids)} PRO players from HLTV")
    return pro_ids


# ============================================================
# 2. GET FPL CONFIGS (ALL REGIONS)
# ============================================================

def get_fpl_configs() -> List[Dict]:
    """Fetch all FPL configuration regions"""
    url = "https://www.faceit.com/api/fpl/v1/configs"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        configs = data.get("payload", [])
        
        # Filter only CS2 configs
        cs2_configs = [c for c in configs if c.get("game") == "cs2"]
        
        print(f"\n🌍 Found {len(cs2_configs)} CS2 FPL regions:")
        for c in cs2_configs:
            print(f"   • {c.get('name')} (ID: {c.get('id')})")
        
        return cs2_configs
    
    except Exception as e:
        print(f"❌ Error fetching FPL configs: {e}")
        return []


# ============================================================
# 3. COLLECT FPL PLAYERS FROM CONFIG
# ============================================================

def collect_fpl_players(config_id: str, season: int = 6) -> Set[str]:
    """Collect all players from a specific FPL config"""
    players = set()
    limit = 50
    offset = 0
    
    while True:
        url = f"https://www.faceit.com/api/fpl/v1/configs/{config_id}/points"
        params = {"season_number": season, "limit": limit, "offset": offset}
        
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                break
            
            data = r.json()
            payload = data.get("payload", [])
            
            if not payload:
                break
            
            for p in payload:
                uid = p.get("user_id")
                if uid:
                    players.add(uid)
            
            # Check if there's more data
            if not data.get("has_more", False):
                break
            
            offset += limit
            time.sleep(0.1)
        
        except Exception as e:
            print(f"   ⚠️  Error at offset {offset}: {e}")
            break
    
    return players


# ============================================================
# 4. COLLECT ALL FPL HIGH-LEVEL PLAYERS
# ============================================================

def collect_all_fpl_players(season: int = 6) -> Set[str]:
    """Collect players from all FPL regions"""
    configs = get_fpl_configs()
    all_fpl = set()
    
    for config in configs:
        config_id = config.get("id")
        region_name = config.get("name", "Unknown")
        
        print(f"\n🔄 Collecting from {region_name}...")
        region_players = collect_fpl_players(config_id, season)
        print(f"   ✅ Found {len(region_players)} players")
        
        all_fpl |= region_players
        time.sleep(0.2)
    
    print(f"\n✅ Total FPL players collected: {len(all_fpl)}")
    return all_fpl


# ============================================================
# 5. COLLECT PLAYERS FROM MATCHMAKING RANKING
# ============================================================

def collect_ranking_players_by_elo(
    region: str = "EU",
    min_elo: int = 500,
    max_elo: int = 1350,
    target_count: int = 700
) -> Set[str]:
    """
    Collect players from matchmaking ranking within ELO range
    Uses binary search approach to find positions efficiently
    """
    players = set()
    limit = 50
    
    print(f"\n🎯 Collecting players with ELO {min_elo}-{max_elo}...")
    
    # Sample positions to find the right ELO range
    # Start from various positions to find our target ELO range
    search_positions = list(range(0, 5091398, 50000))  # Sample every 50k positions
    
    for start_pos in search_positions:
        if len(players) >= target_count:
            break
        
        url = f"https://www.faceit.com/api/ranking/v1/globalranking/cs2/{region}"
        params = {"limit": limit, "position": start_pos}
        
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            
            data = r.json()
            payload = data.get("payload", [])
            
            if not payload:
                continue
            
            # Check if any player in this batch matches our ELO range
            batch_match = False
            for player in payload:
                elo = player.get("elo", 0)
                if min_elo <= elo <= max_elo:
                    batch_match = True
                    uid = player.get("user", {}).get("id")
                    if uid:
                        players.add(uid)
            
            # If we found matches, collect more from nearby positions
            if batch_match:
                print(f"   📍 Found ELO range at position ~{start_pos}, collecting nearby...")
                
                # Collect from nearby positions (±10k)
                for offset in range(max(0, start_pos - 10000), start_pos + 10000, limit):
                    if len(players) >= target_count:
                        break
                    
                    params = {"limit": limit, "position": offset}
                    try:
                        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
                        if r.status_code != 200:
                            continue
                        
                        data = r.json()
                        for player in data.get("payload", []):
                            elo = player.get("elo", 0)
                            if min_elo <= elo <= max_elo:
                                uid = player.get("user", {}).get("id")
                                if uid:
                                    players.add(uid)
                        
                        time.sleep(0.05)
                    except:
                        continue
            
            time.sleep(0.1)
            
            if len(players) % 100 == 0 and len(players) > 0:
                print(f"   💾 Collected {len(players)} players so far...")
        
        except Exception as e:
            continue
    
    print(f"   ✅ Collected {len(players)} players in ELO range {min_elo}-{max_elo}")
    return players


# ============================================================
# 6. COLLECT HIGH-LEVEL PLAYERS FROM TOP RANKING
# ============================================================

def collect_top_ranking_players(region: str = "EU", target: int = 700) -> Set[str]:
    """Collect top players from matchmaking (ELO >= 1751)"""
    players = set()
    limit = 50
    position = 0
    
    print(f"\n🏆 Collecting TOP ranking players (ELO ≥ 1751)...")
    
    while len(players) < target:
        url = f"https://www.faceit.com/api/ranking/v1/globalranking/cs2/{region}"
        params = {"limit": limit, "position": position}
        
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                break
            
            data = r.json()
            payload = data.get("payload", [])
            
            if not payload:
                break
            
            for player in payload:
                elo = player.get("elo", 0)
                if elo >= HIGH_LEVEL_MIN_ELO:
                    uid = player.get("user", {}).get("id")
                    if uid:
                        players.add(uid)
                else:
                    # Once we hit lower ELO, we can stop
                    print(f"   ℹ️  Reached ELO {elo} at position {position}, stopping...")
                    return players
            
            position += limit
            time.sleep(0.05)
            
            if position % 500 == 0:
                print(f"   💾 Collected {len(players)} high-level players...")
        
        except Exception as e:
            print(f"   ⚠️  Error at position {position}: {e}")
            break
    
    print(f"   ✅ Collected {len(players)} high-level players")
    return players


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    # Step 1: Load PRO players
    print("\n" + "="*60)
    print("STEP 1: LOADING PRO PLAYERS")
    print("="*60)
    print("STEP 1: LOADING PRO PLAYERS")
    print("="*60)
    pro_ids = load_pro_ids(config.PRO_DATA_CSV)
    
    # Step 2: Collect FPL players (HIGH-LEVEL)
    print("\n" + "="*60)
    print("STEP 2: COLLECTING FPL HIGH-LEVEL PLAYERS")
    print("="*60)
    fpl_ids = collect_all_fpl_players(season=6)
    
    # Step 3: Collect TOP MM ranking (HIGH-LEVEL)
    print("\n" + "="*60)
    print("STEP 3: COLLECTING TOP MATCHMAKING PLAYERS")
    print("="*60)
    top_mm_ids = collect_top_ranking_players(region="EU", target=TARGET_HIGH_LEVEL)
    
    # Combine and clean HIGH-LEVEL
    high_level_ids = (fpl_ids | top_mm_ids) - pro_ids
    print(f"\n✅ Total HIGH-LEVEL players (excluding PRO): {len(high_level_ids)}")
    
    # Step 4: Collect NORMAL players (ELO 500-1350)
    print("\n" + "="*60)
    print("STEP 4: COLLECTING NORMAL PLAYERS")
    print("="*60)
    normal_ids = collect_ranking_players_by_elo(
        region="EU",
        min_elo=NORMAL_MIN_ELO,
        max_elo=NORMAL_MAX_ELO,
        target_count=TARGET_NORMAL
    )
    
    # Clean NORMAL (remove PRO and HIGH-LEVEL)
    normal_ids = normal_ids - pro_ids - high_level_ids
    print(f"\n✅ Total NORMAL players (clean): {len(normal_ids)}")
    
    # Step 5: Save results
    print("\n" + "="*60)
    print("STEP 5: SAVING RESULTS")
    print("="*60)
    
    with open(config.PRO_IDS_JSON, "w") as f:
        json.dump(list(pro_ids), f, indent=2)
    print(f"✅ Saved {len(pro_ids)} PRO players → {config.PRO_IDS_JSON}")
    
    with open(config.HIGH_LEVEL_IDS_JSON, "w") as f:
        json.dump(list(high_level_ids), f, indent=2)
    print(f"✅ Saved {len(high_level_ids)} HIGH-LEVEL players → {config.HIGH_LEVEL_IDS_JSON}")
    
    with open(config.NORMAL_IDS_JSON, "w") as f:
        json.dump(list(normal_ids), f, indent=2)
    print(f"✅ Saved {len(normal_ids)} NORMAL players → {config.NORMAL_IDS_JSON}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 COLLECTION SUMMARY")
    print("="*60)
    print(f"PRO Players:        {len(pro_ids):>6}")
    print(f"HIGH-LEVEL Players: {len(high_level_ids):>6}")
    print(f"NORMAL Players:     {len(normal_ids):>6}")
    print(f"{'─'*60}")
    print(f"TOTAL:              {len(pro_ids) + len(high_level_ids) + len(normal_ids):>6}")
    print("="*60)
    
    print("\n✅ Data collection complete! Ready for feature extraction.")


if __name__ == "__main__":
    main()

# ╔══════════════════════════════════════════════════════════╗
# ║     FACEIT CS2 PLAYER CLASSIFICATION DATA COLLECTOR     ║
# ║                                                          ║
# ║  Classes:                                                ║
# ║  • PRO: HLTV professional players                        ║
# ║  • HIGH-LEVEL: ELO ≥ 1751 (Level 9-10) + FPL           ║
# ║  • NORMAL: ELO 500-1350 (Level 2-6)                     ║
# ║                                                          ║
# ║  Excluded: Level 7-8 (ELO 1351-1750) for clear gaps     ║
# ╚══════════════════════════════════════════════════════════╝


# ============================================================
# STEP 1: LOADING PRO PLAYERS
# ============================================================
# STEP 1: LOADING PRO PLAYERS
# ============================================================
# ✅ Loaded 577 PRO players from HLTV

# ============================================================
# STEP 2: COLLECTING FPL HIGH-LEVEL PLAYERS
# ============================================================

# 🌍 Found 4 CS2 FPL regions:
#    • FPL Middle East (ID: 67444abee4404b6b2c55f15e)
#    • FPL South America (ID: 67444aae3cbf749cff997b7f)
#    • FPL North America (ID: 67444a673ef06677eeb70194)
#    • FPL Europe (ID: 6744498be4404b6b2c55f15d)

# 🔄 Collecting from FPL Middle East...
#    ✅ Found 105 players

# 🔄 Collecting from FPL South America...
#    ✅ Found 188 players

# 🔄 Collecting from FPL North America...
#    ✅ Found 145 players

# 🔄 Collecting from FPL Europe...
#    ✅ Found 225 players

# ✅ Total FPL players collected: 662

# ============================================================
# STEP 3: COLLECTING TOP MATCHMAKING PLAYERS
# ============================================================

# 🏆 Collecting TOP ranking players (ELO ≥ 1751)...
#    💾 Collected 500 high-level players...
#    ✅ Collected 700 high-level players

# ✅ Total HIGH-LEVEL players (excluding PRO): 796

# ============================================================
# STEP 4: COLLECTING NORMAL PLAYERS
# ============================================================

# 🎯 Collecting players with ELO 500-1350...
#    📍 Found ELO range at position ~1150000, collecting nearby...
#    ✅ Collected 747 players in ELO range 500-1350

# ✅ Total NORMAL players (clean): 747

# ============================================================
# STEP 5: SAVING RESULTS
# ============================================================
# ✅ Saved 577 PRO players → /home/ansinitro/faceit-classifier/data/pro_faceit_ids.json
# ✅ Saved 796 HIGH-LEVEL players → /home/ansinitro/faceit-classifier/data/high_level_faceit_ids.json
# ✅ Saved 747 NORMAL players → /home/ansinitro/faceit-classifier/data/normal_faceit_ids.json

# ============================================================
# 📊 COLLECTION SUMMARY
# ============================================================
# PRO Players:           577
# HIGH-LEVEL Players:    796
# NORMAL Players:        747
# ────────────────────────────────────────────────────────────
# TOTAL:                2120
# ============================================================
