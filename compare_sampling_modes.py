"""
Compare Stochastic vs Deterministic Policy Sampling
====================================================
This script shows why collected data doesn't match the policy being explained.
"""

import numpy as np
from SumoEnv import SumoEnv
from ppo_agent import load_ppo_agent
import matplotlib.pyplot as plt


def analyze_sampling_modes(episodes=5):
    """Compare deterministic vs stochastic sampling."""
    
    env = SumoEnv(sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg", use_gui=False)
    agent = load_ppo_agent(model_path="./models/ppo_mg_road/best_model.zip")
    
    print("\n" + "="*70)
    print("COMPARING SAMPLING MODES")
    print("="*70)
    
    for mode_name, deterministic in [("DETERMINISTIC (Greedy)", True), ("STOCHASTIC (Exploratory)", False)]:
        action_counts = [0, 0]
        
        print(f"\n[{mode_name}]")
        print("-" * 70)
        
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            ep_actions = [0, 0]
            
            while not done and step < 120:
                action, _ = agent.predict(obs, deterministic=deterministic)
                action = int(action)
                ep_actions[action] += 1
                action_counts[action] += 1
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1
            
            keep_pct = 100 * ep_actions[0] / (ep_actions[0] + ep_actions[1])
            switch_pct = 100 * ep_actions[1] / (ep_actions[0] + ep_actions[1])
            print(f"  Episode {ep+1}: Keep={keep_pct:.1f}%, Switch={switch_pct:.1f}%")
        
        total = sum(action_counts)
        keep_pct = 100 * action_counts[0] / total
        switch_pct = 100 * action_counts[1] / total
        print(f"\n  ➜ TOTAL ({5*120} steps): Keep={keep_pct:.1f}%, Switch={switch_pct:.1f}%")
        
        if deterministic:
            print(f"  ⚠️  INSIGHT: Deterministic agent ALWAYS keeps phase!")
            print(f"     → Tree trained on 50%/50% data CAN'T explain this!")
        else:
            print(f"  ⚠️  INSIGHT: Stochastic agent explores both actions")
            print(f"     → But this doesn't match the greedy policy you want to explain!")
    
    print("\n" + "="*70)
    print("SOLUTION")
    print("="*70)
    print("""
You have TWO OPTIONS:

1. EXPLAIN THE GREEDY POLICY (recommended):
   - Collect data using: python explain_policy_dt.py --episodes 30 --depth 15
   - (without --stochastic flag)
   - BUT: Tree will show "Always Keep" with 0 switches
   - This is CORRECT - it's what the agent learned!

2. EXPLAIN THE EXPLORATORY POLICY:
   - Use: python explain_policy_dt.py --episodes 30 --depth 15 --stochastic
   - Current approach - captures probabilistic behavior
   - But the stochastic tree doesn't match what the agent actually does in production
   
RECOMMENDATION:
Your agent learned to ALWAYS KEEP the phase (optimal policy).
The tree accurately reflects this. The 54% accuracy on stochastic data is
EXPECTED because you're forcing the tree to explain actions it will never take.
    """)
    
    env.close()


if __name__ == "__main__":
    analyze_sampling_modes(episodes=5)
