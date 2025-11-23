import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("⏳ Importing AICoach module...")
    from src.analysis.ai_coach import AICoach
    
    print("⏳ Initializing AI Coach (this may take a while to download/load model)...")
    start_time = time.time()
    coach = AICoach()
    
    if not coach.initialized:
        print("❌ AI Coach failed to initialize.")
        sys.exit(1)
        
    print(f"✅ AI Coach initialized in {time.time() - start_time:.2f} seconds")
    
    # Test generation
    print("\n🧪 Testing generation...")
    stats = {
        "kd_ratio": 1.2,
        "win_rate": 55.0,
        "avg_headshots": 45.0,
        "avg_kills": 22.0,
        "survival_rate": 0.4
    }
    
    advice = coach.analyze_player(stats, 8, 1800)
    print("\n🤖 AI Advice:")
    print("-" * 40)
    print(advice)
    print("-" * 40)
    
    if "Strength" in advice and "Weakness" in advice:
        print("\n✅ Verification SUCCESS: Model generated structured advice.")
    else:
        print("\n⚠️ Verification WARNING: Output format might be unexpected.")

except ImportError as e:
    print(f"❌ ImportError: {e}")
    print("Make sure you have installed the dependencies: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
