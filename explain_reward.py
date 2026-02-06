"""
Reward Decomposition Analysis
==============================
Analyzes and visualizes the multi-objective reward contributions from the PPO agent.
Shows the percentage contribution of each reward component (Emergency, Flow, Truck, Car)
to prove the agent is actively optimizing for emergency vehicles.

Usage:
    python explain_reward.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from SumoEnv import SumoEnv
from ppo_agent import load_ppo_agent


def run_reward_analysis_episode(
    agent,
    env: SumoEnv,
    max_steps: int = 500,
    deterministic: bool = True,
) -> dict:
    """
    Run a single episode and collect reward decomposition data.
    
    Args:
        agent: Trained PPO agent
        env: SumoEnv environment
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy
        
    Returns:
        dict with reward components and metrics
    """
    obs, _ = env.reset()
    done = False
    steps = 0
    
    # Lists to store reward components at each step
    flow_rewards = []
    emergency_rewards = []
    truck_rewards = []
    car_rewards = []
    total_rewards = []
    actions = []
    
    print("[EPISODE] Starting reward decomposition episode...")
    
    while not done and steps < max_steps:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Extract reward components from info
        reward_components = info.get('reward_components', {
            'flow': 0.0,
            'emergency': 0.0,
            'truck': 0.0,
            'car': 0.0
        })
        
        flow_rewards.append(reward_components['flow'])
        emergency_rewards.append(reward_components['emergency'])
        truck_rewards.append(reward_components['truck'])
        car_rewards.append(reward_components['car'])
        total_rewards.append(reward)
        actions.append(int(action))
        
        steps += 1
        
        if (steps + 1) % 50 == 0:
            print(f"  Step {steps + 1}/{max_steps}")
    
    print(f"[EPISODE] Completed {steps} steps")
    
    return {
        'steps': steps,
        'flow_rewards': np.array(flow_rewards),
        'emergency_rewards': np.array(emergency_rewards),
        'truck_rewards': np.array(truck_rewards),
        'car_rewards': np.array(car_rewards),
        'total_rewards': np.array(total_rewards),
        'actions': np.array(actions),
    }


def compute_cumulative_rewards(reward_data: dict) -> dict:
    """
    Compute cumulative rewards over the episode.
    
    Args:
        reward_data: Dict with individual reward component arrays
        
    Returns:
        dict with cumulative values
    """
    return {
        'flow_cumulative': np.cumsum(reward_data['flow_rewards']),
        'emergency_cumulative': np.cumsum(reward_data['emergency_rewards']),
        'truck_cumulative': np.cumsum(reward_data['truck_rewards']),
        'car_cumulative': np.cumsum(reward_data['car_rewards']),
        'total_cumulative': np.cumsum(reward_data['total_rewards']),
    }


def analyze_reward_contributions(reward_data: dict, cumulative_data: dict) -> dict:
    """
    Calculate percentage contributions of each reward component.
    
    Args:
        reward_data: Raw reward components
        cumulative_data: Cumulative rewards
        
    Returns:
        dict with contribution metrics
    """
    total_flow = np.sum(reward_data['flow_rewards'])
    total_emergency = np.sum(reward_data['emergency_rewards'])
    total_truck = np.sum(reward_data['truck_rewards'])
    total_car = np.sum(reward_data['car_rewards'])
    total_reward = np.sum(reward_data['total_rewards'])
    
    # Calculate percentages - based on absolute value contribution when total reward is negative
    # This ensures we still show meaningful percentages
    component_sum = abs(total_flow) + abs(total_emergency) + abs(total_truck) + abs(total_car)
    
    if component_sum > 0:
        flow_pct = (abs(total_flow) / component_sum) * 100
        emergency_pct = (abs(total_emergency) / component_sum) * 100
        truck_pct = (abs(total_truck) / component_sum) * 100
        car_pct = (abs(total_car) / component_sum) * 100
    else:
        flow_pct = emergency_pct = truck_pct = car_pct = 0
    
    return {
        'total_flow': float(total_flow),
        'total_emergency': float(total_emergency),
        'total_truck': float(total_truck),
        'total_car': float(total_car),
        'total_reward': float(total_reward),
        'flow_pct': float(flow_pct),
        'emergency_pct': float(emergency_pct),
        'truck_pct': float(truck_pct),
        'car_pct': float(car_pct),
    }


def plot_reward_decomposition(
    cumulative_data: dict,
    contributions: dict,
    output_path: str = "reward_decomposition.png"
):
    """
    Create a stacked area chart showing cumulative reward decomposition.
    
    Args:
        cumulative_data: Cumulative reward components
        contributions: Contribution metrics
        output_path: Path to save the figure
    """
    steps = np.arange(len(cumulative_data['total_cumulative']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # === Plot 1: Stacked Area Chart ===
    # Create legend labels with actual observed percentages
    stack_labels = [
        f"Flow ({contributions['flow_pct']:.1f}%)",
        f"Emergency ({contributions['emergency_pct']:.1f}%)",
        f"Truck ({contributions['truck_pct']:.1f}%)",
        f"Car ({contributions['car_pct']:.1f}%)",
    ]
    ax1.stackplot(
        steps,
        cumulative_data['flow_cumulative'],
        cumulative_data['emergency_cumulative'],
        cumulative_data['truck_cumulative'],
        cumulative_data['car_cumulative'],
        labels=stack_labels,
        colors=['#1f77b4', '#d62728', '#2ca02c', '#a0a0a0'],  # Blue, Red, Green, Gray
        alpha=0.8
    )
    
    ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Cumulative Reward Decomposition: Agent Priorities', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Pie Chart - Contribution Distribution ===
    sizes = [
        contributions['flow_pct'],
        contributions['emergency_pct'],
        contributions['truck_pct'],
        contributions['car_pct'],
    ]
    labels = [
        f"Flow\n({contributions['flow_pct']:.1f}%)",
        f"Emergency\n({contributions['emergency_pct']:.1f}%)",
        f"Truck\n({contributions['truck_pct']:.1f}%)",
        f"Car\n({contributions['car_pct']:.1f}%)",
    ]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#a0a0a0']
    
    # Only plot pie chart if there's actual data to show
    if sum(sizes) > 0:
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11, 'weight': 'bold'},
                wedgeprops={'alpha': 0.8})
    else:
        # If no meaningful data, show a message
        ax2.text(0.5, 0.5, 'No reward components\nto display', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    ax2.set_title('Total Reward Contribution by Component', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVE] Reward decomposition chart saved to: {output_path}")
    plt.show()


def print_reward_analysis(contributions: dict, reward_data: dict):
    """
    Print detailed reward analysis to console.
    
    Args:
        contributions: Contribution metrics
        reward_data: Raw reward data
    """
    print("\n" + "=" * 80)
    print("REWARD DECOMPOSITION ANALYSIS")
    print("=" * 80)
    
    print(f"\n[TOTALS]")
    print(f"  Total Reward:        {contributions['total_reward']:>10.2f}")
    print(f"  Total Flow Reward:   {contributions['total_flow']:>10.2f}")
    print(f"  Total Emergency:     {contributions['total_emergency']:>10.2f}")
    print(f"  Total Truck:         {contributions['total_truck']:>10.2f}")
    print(f"  Total Car:           {contributions['total_car']:>10.2f}")
    
    print(f"\n[PERCENTAGE CONTRIBUTIONS]")
    print(f"  Flow Component:      {contributions['flow_pct']:>10.1f}%")
    print(f"  Emergency Component: {contributions['emergency_pct']:>10.1f}%")
    print(f"  Truck Component:     {contributions['truck_pct']:>10.1f}%")
    print(f"  Car Component:       {contributions['car_pct']:>10.1f}%")
    
    # Calculate priority evidence
    print(f"\n[PRIORITY EVIDENCE]")
    if contributions['emergency_pct'] > 0:
        print(f"  ✓ Emergency vehicles accounted for {contributions['emergency_pct']:.1f}% of total reward")
        print(f"    (Configured weight: 30% of reward function)")
        if contributions['emergency_pct'] > 15:
            print(f"    → Agent IS actively optimizing for emergency vehicles! ✓")
        else:
            print(f"    → Emergency optimization may need tuning")
    else:
        print(f"  ✗ No emergency vehicles in this episode")
    
    print(f"\n[STATISTICS]")
    print(f"  Average Step Reward: {np.mean(reward_data['total_rewards']):.4f}")
    print(f"  Std Dev Step Reward: {np.std(reward_data['total_rewards']):.4f}")
    print(f"  Min Step Reward:     {np.min(reward_data['total_rewards']):.4f}")
    print(f"  Max Step Reward:     {np.max(reward_data['total_rewards']):.4f}")
    print(f"  Total Steps:         {len(reward_data['total_rewards'])}")
    
    print("\n" + "=" * 80)


def save_analysis_json(contributions: dict, output_path: str = "reward_analysis.json"):
    """
    Save analysis results to JSON for further processing.
    
    Args:
        contributions: Analysis metrics
        output_path: Path to save JSON
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'contributions': contributions,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[SAVE] Analysis saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("REWARD DECOMPOSITION ANALYZER")
    print("=" * 80)
    
    # === Setup ===
    print("\n[SETUP] Loading environment and agent...")
    env = SumoEnv(sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg", use_gui=False)
    
    model_path = "./models/ppo_mg_road/best_model.zip"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("        Please train the model first: python train_ppo.py")
        return
    
    agent = load_ppo_agent(model_path=model_path)
    print(f"[SETUP] Loaded agent from: {model_path}")
    
    # === Run Episode with Decomposition ===
    # Keep running episodes until we find one with emergency vehicles (or reach max attempts)
    print("\n[COLLECTION] Running episodes to find one with emergency vehicles...")
    reward_data = None
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"\n[ATTEMPT {attempt}/{max_attempts}] Running episode...")
        reward_data = run_reward_analysis_episode(
            agent, 
            env, 
            max_steps=2000,  # Increased from 500 to 2000 (200 minutes of simulation)
            deterministic=True
        )
        
        # Check if we got emergency vehicles
        if reward_data['emergency_rewards'].sum() != 0:
            print(f"[SUCCESS] Found emergency vehicles in episode {attempt}!")
            break
        else:
            print(f"[INFO] No emergency vehicles in this episode. Trying again...")
    
    if reward_data is None:
        print("[ERROR] Could not collect episode data")
        return
    
    # === Compute Cumulative Rewards ===
    print("[ANALYSIS] Computing cumulative rewards...")
    cumulative_data = compute_cumulative_rewards(reward_data)
    
    # === Analyze Contributions ===
    print("[ANALYSIS] Analyzing reward contributions...")
    contributions = analyze_reward_contributions(reward_data, cumulative_data)
    
    # === Print Results ===
    print_reward_analysis(contributions, reward_data)
    
    # === Visualize ===
    print("\n[VISUALIZATION] Creating plots...")
    plot_reward_decomposition(
        cumulative_data,
        contributions,
        output_path="reward_decomposition.png"
    )
    
    # === Save Results ===
    save_analysis_json(contributions, output_path="reward_analysis.json")
    
    print("\n[COMPLETE] Reward decomposition analysis finished!")


if __name__ == "__main__":
    main()
