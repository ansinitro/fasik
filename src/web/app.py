from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import joblib
import json
from tensorflow import keras
import threading
import sys
import os


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import AI Coach (optional feature)
try:
    from src.analysis.ai_coach import AICoach
    AI_COACH_AVAILABLE = True
except Exception as e:
    print(f"⚠️  AI Coach not available: {e}")
    AICoach = None
    AI_COACH_AVAILABLE = False

# Initialize AI Coach in background
ai_coach = None
def init_ai_coach():
    global ai_coach
    if AI_COACH_AVAILABLE and AICoach is not None:
        try:
            ai_coach = AICoach()
        except Exception as e:
            print(f"⚠️  Failed to initialize AI Coach: {e}")

if AI_COACH_AVAILABLE:
    threading.Thread(target=init_ai_coach, daemon=True).start()

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

FACEIT_API_KEY = "af872248-2c8a-40ae-b808-307b1394d9b2"
GAME_ID = "cs2"

HEADERS = {
    "Authorization": f"Bearer {FACEIT_API_KEY}",
    "Accept": "application/json"
}

BASE_URL = "https://open.faceit.com/data/v4"

# Load models and preprocessors at startup
print("🔄 Loading models and preprocessors...")

try:
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    with open("data/feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    with open("models/best_model.txt", "r") as f:
        best_model_name = f.read().strip()
    
    # Load best model
    if best_model_name == "Neural Network":
        model = keras.models.load_model("models/neural_network.h5")
    elif best_model_name == "XGBoost":
        model = joblib.load("models/xgboost.pkl")
    elif best_model_name == "Random Forest":
        model = joblib.load("models/random_forest.pkl")
    elif best_model_name == "Ensemble Voting":
        model = joblib.load("models/ensemble_voting.pkl")
    else:
        model = joblib.load("models/logistic_regression.pkl")
    
    # Load class statistics for analysis
    with open("data/class_statistics.json", "r") as f:
        class_stats = json.load(f)
    
    print(f"✅ Loaded {best_model_name} model")
    print("✅ All components loaded successfully!")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_player_by_nickname(nickname):
    """Fetch player ID by nickname"""
    url = f"{BASE_URL}/players"
    params = {"nickname": nickname}
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_player_stats(player_id):
    """Fetch player statistics"""
    url = f"{BASE_URL}/players/{player_id}/stats/{GAME_ID}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def extract_features_from_stats(player_data, stats_data):
    """Extract features from player data - matches features.py logic"""
    try:
        # Basic info from profile
        games = player_data.get("games", {}).get(GAME_ID, {})
        skill_level = games.get("skill_level", 0)
        faceit_elo = games.get("faceit_elo", 0)
        
        # Get lifetime stats
        lifetime = stats_data.get("lifetime", {})
        if not lifetime:
            return None
        
        # Helper function to safely get stat value from lifetime
        def get_lifetime_stat(stat_name, default=0):
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
            matches = 1
        
        # Get segments (map-specific stats)
        segments = stats_data.get("segments", [])
        
        # Aggregate stats from segments
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
        
        # Build feature dictionary matching features.py
        features = {
            # Profile stats
            "skill_level": skill_level,
            "faceit_elo": faceit_elo,
            
            # Lifetime stats
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
            
            # Additional stats from lifetime
            "entry_success_rate": get_lifetime_stat("Entry Success Rate"),
            "clutch_1v1_win_rate": get_lifetime_stat("1v1 Win Rate"),
            "clutch_1v2_win_rate": get_lifetime_stat("1v2 Win Rate"),
            "flash_success_rate": get_lifetime_stat("Flash Success Rate"),
            "utility_damage_per_round": get_lifetime_stat("Utility Damage per Round"),
        }
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_player(features, predicted_class, probabilities):
    """Analyze player strengths and weaknesses"""
    analysis = {
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }
    
    # Get class statistics
    class_name = label_encoder.classes_[predicted_class]
    class_avg = class_stats[class_name]
    
    # Define important metrics
    metrics_to_check = {
        "kd_ratio": "K/D Ratio",
        "win_rate": "Win Rate",
        "avg_headshots": "Headshot %",
        "kr_ratio": "K/R Ratio",
        "impact_score": "Impact Score",
        "survival_rate": "Survival Rate",
    }
    
    for feature, display_name in metrics_to_check.items():
        if feature in features and feature in class_avg:
            player_value = features[feature]
            avg_value = class_avg[feature]["mean"]
            std_value = class_avg[feature]["std"]
            
            # Check if strength
            if player_value > avg_value + std_value:
                analysis["strengths"].append({
                    "metric": display_name,
                    "value": round(player_value, 2),
                    "class_avg": round(avg_value, 2),
                    "percentile": "Top 16%"
                })
            
            # Check if weakness
            elif player_value < avg_value - std_value:
                analysis["weaknesses"].append({
                    "metric": display_name,
                    "value": round(player_value, 2),
                    "class_avg": round(avg_value, 2),
                    "percentile": "Bottom 16%"
                })
    
    # Generate recommendations
    if len(analysis["weaknesses"]) > 0:
        for weakness in analysis["weaknesses"]:
            if "K/D" in weakness["metric"]:
                analysis["recommendations"].append("Focus on positioning and crosshair placement to improve your K/D ratio")
            elif "Win Rate" in weakness["metric"]:
                analysis["recommendations"].append("Work on team communication and strategic play to increase win rate")
            elif "Headshot" in weakness["metric"]:
                analysis["recommendations"].append("Practice aim training in deathmatch to improve headshot accuracy")
            elif "Survival" in weakness["metric"]:
                analysis["recommendations"].append("Work on game sense and positioning to die less frequently")
    else:
        analysis["recommendations"].append("Great performance! Focus on consistency and adapting to different playstyles")
    
    return analysis


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    """Render home page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict player class and analyze performance"""
    try:
        data = request.get_json()
        nickname = data.get("nickname", "").strip()
        
        if not nickname:
            return jsonify({"error": "Please provide a nickname"}), 400
        
        # Get player data
        player_data = get_player_by_nickname(nickname)
        if not player_data:
            return jsonify({"error": f"Player '{nickname}' not found on FACEIT"}), 404
        
        player_id = player_data["player_id"]
        
        # Get player stats
        stats_data = get_player_stats(player_id)
        if not stats_data:
            return jsonify({"error": "Could not fetch player statistics"}), 404
        
        # Extract features
        features = extract_features_from_stats(player_data, stats_data)
        if not features:
            return jsonify({"error": "Could not extract features from player data"}), 500
        
        # Prepare features for prediction
        feature_vector = [features.get(fn, 0) for fn in feature_names]
        feature_vector_scaled = scaler.transform([feature_vector])
        
        # Make prediction
        if best_model_name == "Neural Network":
            probas = model.predict(feature_vector_scaled, verbose=0)[0]
            predicted_class = np.argmax(probas)
        else:
            probas = model.predict_proba(feature_vector_scaled)[0]
            predicted_class = model.predict(feature_vector_scaled)[0]
        
        # Convert ALL numpy types to Python native types
        predicted_class = int(predicted_class)
        probas = [float(p) for p in probas]
        
        # Get class names and probabilities
        class_probs = {
            str(label_encoder.classes_[i]): float(probas[i] * 100)
            for i in range(len(label_encoder.classes_))
        }
        
        predicted_class_name = str(label_encoder.classes_[predicted_class])
        
        # Analyze player (Rule-based)
        analysis = analyze_player(features, predicted_class, probas)
        
        # AI Coach Analysis
        ai_advice = "AI Coach is still loading..."
        if ai_coach and ai_coach.initialized:
            ai_advice = ai_coach.analyze_player(
                features, 
                int(float(features["skill_level"])), 
                int(float(features["faceit_elo"]))
            )
        
        # Helper to convert numpy types recursively
        def to_python_type(obj):
            """Convert numpy/pandas types to Python native types"""
            if isinstance(obj, dict):
                return {k: to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_python_type(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return to_python_type(obj.tolist())
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        
        # Build response with explicit type conversion
        response = {
            "player": {
                "nickname": str(player_data["nickname"]),
                "country": str(player_data.get("country", "Unknown")),
                "elo": int(float(features["faceit_elo"])),
                "level": int(float(features["skill_level"])),
            },
            "prediction": {
                "class": predicted_class_name,
                "confidence": float(probas[predicted_class] * 100),
                "probabilities": to_python_type(class_probs)
            },
            "statistics": {
                "K/D Ratio": float(features["kd_ratio"]),
                "Win Rate": float(features["win_rate"]),
                "Headshot %": float(features["avg_headshots"]),
                "Matches Played": int(float(features["matches"])),
                "Average Kills": float(features["avg_kills"]),
            },
            "analysis": to_python_type(analysis),
            "ai_analysis": ai_advice
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["GET"])
def search_players():
    """Search players via FACEIT API (proxy to avoid CORS)"""
    query = request.args.get("query", "").strip()
    
    if not query or len(query) < 2:
        return jsonify({"players": {"results": []}})
    
    try:
        # Call FACEIT search API
        url = f"https://www.faceit.com/api/searcher/v1/all"
        params = {"limit": 10, "query": query}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Filter only CS2 players
            players = data.get("players", {}).get("results", [])
            cs2_players = [p for p in players if any(g.get("name") == "cs2" for g in p.get("games", []))]
            return jsonify({"players": {"results": cs2_players[:10]}})
        
        return jsonify({"players": {"results": []}})
    
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"players": {"results": []}})


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": best_model_name})


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)