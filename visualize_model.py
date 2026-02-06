"""
Visualize PPO Agent Behavior
=============================
This script checks if your agent is stable or "flickering" (switching actions rapidly).
A stable agent should maintain consistent actions, while a flickering agent switches constantly.

Usage:
    python visualize_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from SumoEnv import SumoEnv
from ppo_agent import load_ppo_agent


def visualize_agent_behavior(episodes=3, max_steps=500):
    """
    Visualize agent behavior over episodes.
    
    Args:
        episodes: Number of episodes to visualize
        max_steps: Max steps per episode
    """
    # Load environment and agent
    env = SumoEnv(sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg", use_gui=False)
    agent = load_ppo_agent(model_path="./models/ppo_mg_road/best_model.zip")
    
    fig, axes = plt.subplots(episodes, 1, figsize=(14, 4 * episodes))
    if episodes == 1:
        axes = [axes]
    
    all_action_counts = []
    all_switch_counts = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        actions = []
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Use deterministic (greedy) policy
            action, _ = agent.predict(obs, deterministic=True)
            actions.append(int(action))
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        
        # Count statistics
        actions = np.array(actions)
        n_keep = np.sum(actions == 0)
        n_switch = np.sum(actions == 1)
        n_switches = np.sum(np.diff(actions) != 0)  # How many times action changed
        
        all_action_counts.append((n_keep, n_switch))
        all_switch_counts.append(n_switches)
        
        # Plot action sequence
        ax = axes[ep]
        ax.plot(actions, marker='o', linewidth=2, markersize=4, color='steelblue')
        ax.set_ylabel('Action (0=Keep, 1=Switch)', fontsize=11)
        ax.set_title(
            f'Episode {ep + 1}: Keep={n_keep}, Switch={n_switch}, '
            f'Actual Switches={n_switches} ({100*n_switches/len(actions):.1f}%)',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([0, 1])
    
    plt.xlabel('Step', fontsize=11)
    plt.tight_layout()
    plt.savefig('./logs/policy_explanation/agent_behavior_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to ./logs/policy_explanation/agent_behavior_visualization.png")
    
    # Statistical summary
    print("\n" + "="*70)
    print("AGENT BEHAVIOR ANALYSIS")
    print("="*70)
    
    for ep, ((n_keep, n_switch), n_switches) in enumerate(zip(all_action_counts, all_switch_counts)):
        switch_rate = 100 * n_switches / (n_keep + n_switch)
        print(f"\nEpisode {ep + 1}:")
        print(f"  - Keep Phase: {n_keep} steps ({100*n_keep/(n_keep+n_switch):.1f}%)")
        print(f"  - Switch Phase: {n_switch} steps ({100*n_switch/(n_keep+n_switch):.1f}%)")
        print(f"  - Actual Switches: {n_switches} times ({switch_rate:.1f}%)")
        
        if switch_rate > 30:
            print(f"  ⚠️  WARNING: High switch rate (>{30}%) - Agent might be FLICKERING!")
        elif switch_rate < 5:
            print(f"  ✓ GOOD: Low switch rate (<5%) - Agent is STABLE")
        else:
            print(f"  ~ MODERATE: Agent switches periodically")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print("""
If your agent is FLICKERING (high switch rate):
  → The tree will struggle because the agent's behavior is chaotic
  → You may want to add regularization or emergency vehicle awareness
  
If your agent is STABLE (low switch rate):
  → The tree should learn well with increased depth
  → 54% accuracy suggests the tree depth was the issue
    """)
    
    env.close()


if __name__ == "__main__":
    visualize_agent_behavior(episodes=3, max_steps=500)
