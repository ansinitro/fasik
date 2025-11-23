"""
Simple Rule-Based AI Coach for CS2 Player Analysis
Provides intelligent insights without requiring transformers/torch
"""

class SimpleAICoach:
    """Lightweight AI Coach using rule-based analysis"""
    
    def __init__(self):
        self.initialized = True
        print("✅ Simple AI Coach initialized!")
    
    def analyze_player(self, stats, level, elo):
        """Generate intelligent advice based on player stats"""
        try:
            kd = stats.get('kd_ratio', 0)
            win_rate = stats.get('win_rate', 0)
            hs_pct = stats.get('avg_headshots', 0)
            avg_kills = stats.get('avg_kills', 0)
            
            # Determine player tier
            if elo >= 3000:
                tier = "Elite"
                tier_desc = "top-tier"
            elif elo >= 2000:
                tier = "Advanced"
                tier_desc = "high-level"
            elif elo >= 1500:
                tier = "Intermediate"
                tier_desc = "mid-level"
            else:
                tier = "Developing"
                tier_desc = "entry-level"
            
            analysis = []
            
            # Header
            analysis.append(f"🎯 AI COACH ANALYSIS - {tier} Player (Level {level}, {elo} ELO)")
            analysis.append("")
            
            # Strength Analysis
            analysis.append("💪 STRENGTHS:")
            if kd >= 1.3:
                analysis.append(f"• Excellent fragging power (K/D: {kd:.2f}) - You consistently win gunfights")
            elif kd >= 1.1:
                analysis.append(f"• Solid fragging ability (K/D: {kd:.2f}) - Good combat effectiveness")
            
            if win_rate >= 55:
                analysis.append(f"• Strong win rate ({win_rate:.1f}%) - You contribute to team victories")
            
            if hs_pct >= 50:
                analysis.append(f"• Exceptional aim precision ({hs_pct:.1f}% HS) - Your crosshair placement is excellent")
            elif hs_pct >= 45:
                analysis.append(f"• Good aim control ({hs_pct:.1f}% HS) - Solid mechanical skills")
            
            if not any("•" in line for line in analysis[-3:]):
                analysis.append("• Consistent performance across matches - Keep building on your fundamentals")
            
            analysis.append("")
            
            # Weakness Analysis
            analysis.append("📊 AREAS FOR GROWTH:")
            
            if kd < 1.0:
                analysis.append(f"• K/D ratio ({kd:.2f}) needs improvement - Focus on staying alive longer")
                analysis.append("  → Practice crosshair placement and pre-aiming common angles")
                analysis.append("  → Review your deaths to identify positioning mistakes")
            elif kd < 1.15:
                analysis.append(f"• K/D ratio ({kd:.2f}) has room to grow - Work on trading kills efficiently")
            
            if win_rate < 50:
                analysis.append(f"• Win rate ({win_rate:.1f}%) below 50% - Improve team coordination")
                analysis.append("  → Focus on communication and playing for the team")
                analysis.append("  → Learn when to save vs. force buy")
            elif win_rate < 53:
                analysis.append(f"• Win rate ({win_rate:.1f}%) could be higher - Work on clutch situations")
            
            if hs_pct < 40:
                analysis.append(f"• Headshot % ({hs_pct:.1f}%) needs work - Improve aim precision")
                analysis.append("  → Practice aim training maps (Aim Botz, Refrag Arena)")
                analysis.append("  → Focus on head-level crosshair placement")
            elif hs_pct < 45:
                analysis.append(f"• Headshot % ({hs_pct:.1f}%) can improve - Refine your aim")
            
            if not any("•" in line for line in analysis[-5:]):
                analysis.append("• Minor refinements needed - You're performing well overall")
            
            analysis.append("")
            
            # Personalized Training Plan
            analysis.append("🎓 PERSONALIZED TRAINING PLAN:")
            
            if kd < 1.0:
                analysis.append("1. Survival First: Focus on not dying. Play safer angles and use utility")
                analysis.append("2. Aim Training: 15 min daily on aim maps before competitive")
            elif kd < 1.2:
                analysis.append("1. Aggressive Plays: Take more calculated risks to increase impact")
                analysis.append("2. Entry Fragging: Practice being first in and winning duels")
            else:
                analysis.append("1. Consistency: Maintain your high level across all maps")
                analysis.append("2. Leadership: Help teammates with calls and strat execution")
            
            if hs_pct < 45:
                analysis.append("3. Crosshair Discipline: Always keep crosshair at head level")
                analysis.append("4. Spray Control: Master first 10 bullets of AK/M4")
            else:
                analysis.append("3. Movement: Work on counter-strafing and peeking mechanics")
                analysis.append("4. Game Sense: Study pro demos to learn positioning")
            
            if win_rate < 52:
                analysis.append("5. Team Play: Focus on trading kills and supporting teammates")
            else:
                analysis.append("5. Clutch Practice: Train 1vX scenarios in retake servers")
            
            analysis.append("")
            
            # Next Level Goals
            if tier == "Developing":
                analysis.append("🎯 NEXT MILESTONE: Reach 1500 ELO (Intermediate)")
                analysis.append("   Focus: Fundamentals, aim, and game sense")
            elif tier == "Intermediate":
                analysis.append("🎯 NEXT MILESTONE: Reach 2000 ELO (Advanced)")
                analysis.append("   Focus: Consistency, utility usage, and team coordination")
            elif tier == "Advanced":
                analysis.append("🎯 NEXT MILESTONE: Reach 3000 ELO (Elite)")
                analysis.append("   Focus: Advanced tactics, leadership, and clutch ability")
            else:
                analysis.append("🎯 NEXT MILESTONE: Maintain Elite status and compete")
                analysis.append("   Focus: Tournament play and team synergy")
            
            return "\n".join(analysis)
            
        except Exception as e:
            print(f"❌ AI Analysis error: {e}")
            return "AI Coach analysis temporarily unavailable. Please try again."


# For backward compatibility
AICoach = SimpleAICoach
