import pandas as pd
import numpy as np
import json

print("""
╔══════════════════════════════════════════════════════════╗
║         CALCULATE CLASS STATISTICS FOR ANALYSIS         ║
╚══════════════════════════════════════════════════════════╝
""")

# Load the dataset
df = pd.read_csv("./data/faceit_players_dataset.csv")

print(f"✅ Loaded dataset with {len(df)} players")
print(f"\n📋 Class distribution:")
print(df["player_class"].value_counts())

# Calculate statistics for each class
class_stats = {}

for player_class in df["player_class"].unique():
    class_df = df[df["player_class"] == player_class]
    
    class_stats[player_class] = {}
    
    # Calculate mean and std for numeric columns
    numeric_cols = class_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ["player_id"]:
            class_stats[player_class][col] = {
                "mean": float(class_df[col].mean()),
                "std": float(class_df[col].std()),
                "min": float(class_df[col].min()),
                "max": float(class_df[col].max()),
                "median": float(class_df[col].median())
            }
    
    print(f"\n✅ Calculated statistics for {player_class} class")

# Save class statistics
with open("data/class_statistics.json", "w") as f:
    json.dump(class_stats, f, indent=2)

print(f"\n💾 Saved class statistics to: data/class_statistics.json")

# Display sample statistics
print("\n" + "="*60)
print("SAMPLE CLASS STATISTICS")
print("="*60)

for player_class in class_stats.keys():
    print(f"\n{player_class}:")
    print(f"  K/D Ratio:  {class_stats[player_class]['kd_ratio']['mean']:.2f} ± {class_stats[player_class]['kd_ratio']['std']:.2f}")
    print(f"  Win Rate:   {class_stats[player_class]['win_rate']['mean']:.2f}% ± {class_stats[player_class]['win_rate']['std']:.2f}%")
    print(f"  Headshot %: {class_stats[player_class]['avg_headshots']['mean']:.2f}% ± {class_stats[player_class]['avg_headshots']['std']:.2f}%")
    print(f"  ELO:        {class_stats[player_class]['faceit_elo']['mean']:.0f} ± {class_stats[player_class]['faceit_elo']['std']:.0f}")

print("\n✅ Class statistics calculation complete!")
