from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import joblib
import json
from tensorflow import keras

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
    """Extract features from player data"""
    try:
        games = player_data.get("games", {}).get(GAME_ID, {})
        skill_level = games.get("skill_level", 0)
        faceit_elo = games.get("faceit_elo", 0)
        
        segments = stats_data.get("segments", [])
        overall_stats = {}
        
        for segment in segments:
            if segment.get("type") == "Overall":
                overall_stats = segment.get("stats", {})
                break
        
        def get_stat(stat_name, default=0):
            stat_obj = overall_stats.get(stat_name, {})
            if isinstance(stat_obj, dict):
                return float(stat_obj.get("value", default))
            return default
        
        matches = get_stat("Matches")
        if matches == 0:
            matches = 1
        
        # Base features
        features = {
            "skill_level": skill_level,
            "faceit_elo": faceit_elo,
            "matches": matches,
            "kd_ratio": get_stat("Average K/D Ratio"),
            "kr_ratio": get_stat("Average K/R Ratio"),
            "win_rate": get_stat("Win Rate %"),
            "avg_kills": get_stat("Average Kills"),
            "avg_deaths": get_stat("Average Deaths"),
            "avg_assists": get_stat("Average Assists"),
            "avg_headshots": get_stat("Average Headshots %"),
            "avg_mvps": get_stat("Average MVPs"),
            "triple_kills": get_stat("Triple Kills"),
            "quadro_kills": get_stat("Quadro Kills"),
            "penta_kills": get_stat("Penta Kills"),
            "current_win_streak": get_stat("Current Win Streak"),
            "longest_win_streak": get_stat("Longest Win Streak"),
            "total_kills": get_stat("Total Kills"),
            "total_deaths": get_stat("Total Deaths"),
            "total_assists": get_stat("Assists"),
        }
        
        # Derived features
        features["kills_per_match"] = features["total_kills"] / matches
        features["deaths_per_match"] = features["total_deaths"] / matches
        features["assists_per_match"] = features["total_assists"] / matches
        
        # Recent win rate
        lifetime = stats_data.get("lifetime", {})
        recent_results = lifetime.get("Recent Results", [])
        if len(recent_results) > 0:
            recent_wins = sum(1 for r in recent_results if r == "1")
            features["recent_win_rate"] = (recent_wins / len(recent_results)) * 100
        else:
            features["recent_win_rate"] = features["win_rate"]
        
        # Engineered features
        features["kill_efficiency"] = features["total_kills"] / matches
        features["survival_rate"] = 1 - (features["total_deaths"] / matches)
        features["impact_score"] = (features["avg_kills"] + features["avg_assists"] * 0.5) - features["avg_deaths"]
        features["win_contribution"] = features["win_rate"] * features["kd_ratio"]
        features["special_kills_total"] = features["triple_kills"] + features["quadro_kills"] * 2 + features["penta_kills"] * 3
        features["special_kills_rate"] = features["special_kills_total"] / matches
        features["consistency"] = features["recent_win_rate"] - features["win_rate"]
        features["headshot_skill"] = features["avg_headshots"] * features["kd_ratio"]
        features["experience_level"] = np.log1p(matches)
        features["mvp_rate"] = features["avg_mvps"] / matches
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
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
        
        # Convert numpy types to Python types for JSON serialization
        predicted_class = int(predicted_class)
        probas = [float(p) for p in probas]
        
        # Get class names and probabilities
        class_probs = {
            label_encoder.classes_[i]: float(probas[i] * 100)
            for i in range(len(label_encoder.classes_))
        }
        
        predicted_class_name = label_encoder.classes_[predicted_class]
        
        # Analyze player
        analysis = analyze_player(features, predicted_class, probas)
        
        # Convert all numpy/float32 to Python native types
        def convert_to_native(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Prepare response with type conversion
        response = {
            "player": {
                "nickname": str(player_data["nickname"]),
                "country": str(player_data.get("country", "Unknown")),
                "elo": int(features["faceit_elo"]),
                "level": int(features["skill_level"]),
            },
            "prediction": {
                "class": str(predicted_class_name),
                "confidence": float(round(probas[predicted_class] * 100, 2)),
                "probabilities": convert_to_native(class_probs)
            },
            "statistics": {
                "K/D Ratio": float(round(features["kd_ratio"], 2)),
                "Win Rate": float(round(features["win_rate"], 2)),
                "Headshot %": float(round(features["avg_headshots"], 2)),
                "Matches Played": int(features["matches"]),
                "Average Kills": float(round(features["avg_kills"], 2)),
            },
            "analysis": convert_to_native(analysis)
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": best_model_name})


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)