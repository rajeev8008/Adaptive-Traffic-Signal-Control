"""
Policy Distillation using Decision Tree
========================================
This script approximates the trained PPO agent's behavior using a Decision Tree
to make the neural network's decision logic interpretable.

Features:
- Collects observations and actions from the trained agent
- Trains a simple Decision Tree to mimic the agent's behavior
- Calculates fidelity (how well the tree matches the agent)
- Visualizes the decision tree for human interpretation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Stable-Baselines3 and sklearn
from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Local imports
from SumoEnv import SumoEnv
from ppo_agent import load_ppo_agent


# --- Configuration ---
class ExplainConfig:
    """Configuration for policy distillation"""
    MODEL_PATH = "./models/ppo_mg_road/best_model.zip"
    SUMOCFG_FILE = "SUMO_Trinity_Traffic_sim/osm.sumocfg"
    USE_GUI = False
    
    # Data collection
    COLLECTION_EPISODES = 10
    COLLECTION_SEED = 42
    
    # Decision Tree hyperparameters
    TREE_MAX_DEPTH = 4
    TREE_MIN_SAMPLES_SPLIT = 5
    TREE_MIN_SAMPLES_LEAF = 2
    
    # Output
    OUTPUT_DIR = "./logs/policy_explanation"
    TREE_PNG = "decision_tree_logic.png"
    TREE_SUMMARY = "tree_summary.json"


# --- Feature Names ---
def get_feature_names():
    """
    Generate feature names for the 43-dimensional observation space.
    
    Observation structure:
    - Indices 0-13: Queue lengths per lane (14 lanes)
    - Index 14: Current traffic light phase (1 feature)
    - Indices 15-28: Emergency vehicle counts per lane (14 lanes)
    - Indices 29-42: Bus vehicle counts per lane (14 lanes)
    """
    feature_names = []
    
    # Queue lengths (0-13)
    for i in range(14):
        feature_names.append(f"Queue_Lane_{i}")
    
    # Traffic light phase (14)
    feature_names.append("Traffic_Light_Phase")
    
    # Emergency vehicle counts (15-28)
    for i in range(14):
        feature_names.append(f"Emergency_Lane_{i}")
    
    # Bus vehicle counts (29-42)
    for i in range(14):
        feature_names.append(f"Bus_Lane_{i}")
    
    assert len(feature_names) == 43, f"Expected 43 features, got {len(feature_names)}"
    return feature_names


# --- Data Collection Function ---
def collect_data(agent, env, episodes=10, max_steps=None, stochastic=False):
    """
    Collect observations and actions from the agent running in the environment.
    
    Args:
        agent: Trained PPO agent
        env: SumoEnv instance
        episodes: Number of episodes to collect data from
        max_steps: Max steps per episode (default: env.max_episode_steps)
        stochastic: If True, sample actions from probability distribution (more diverse)
    
    Returns:
        X: numpy array of shape (n_samples, 43) - observations
        y: numpy array of shape (n_samples,) - actions (0 or 1)
    """
    X = []
    y = []
    action_probs = []  # To track probabilities
    
    if max_steps is None:
        max_steps = env.max_episode_steps if hasattr(env, 'max_episode_steps') else 120
    
    print(f"\n[COLLECTION] Collecting data from {episodes} episodes...")
    print(f"[COLLECTION] Max steps per episode: {max_steps}")
    print(f"[COLLECTION] Sampling mode: {'STOCHASTIC (probabilistic)' if stochastic else 'DETERMINISTIC (greedy)'}")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_samples = 0
        
        while not done and step < max_steps:
            # Get action from agent
            if stochastic:
                # Sample from the policy distribution (more diverse)
                action, _ = agent.predict(obs, deterministic=False)
            else:
                # Greedy policy (deterministic)
                action, _ = agent.predict(obs, deterministic=True)
            
            # Store observation and action
            X.append(obs.copy() if isinstance(obs, np.ndarray) else np.array(obs))
            y.append(int(action))
            episode_samples += 1
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        
        print(f"  Episode {episode + 1}/{episodes}: {episode_samples} samples collected")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    action_counts = np.bincount(y)
    print(f"\n[COLLECTION] Total samples collected: {len(X)}")
    print(f"[COLLECTION] Action distribution:")
    print(f"  - Action 0 (Keep Phase): {action_counts[0] if len(action_counts) > 0 else 0} ({100 * action_counts[0] / len(X):.1f}%)")
    print(f"  - Action 1 (Switch Phase): {action_counts[1] if len(action_counts) > 1 else 0} ({100 * (action_counts[1] if len(action_counts) > 1 else 0) / len(X):.1f}%)")
    print(f"[COLLECTION] Shape: X={X.shape}, y={y.shape}")
    
    return X, y


# --- Decision Tree Training ---
def train_decision_tree(X, y, feature_names, max_depth=4):
    """
    Train a Decision Tree classifier to approximate the PPO agent's policy.
    
    Args:
        X: Training observations (n_samples, 43)
        y: Training actions (n_samples,)
        feature_names: List of 43 feature names
        max_depth: Maximum tree depth
    
    Returns:
        dt_classifier: Trained DecisionTreeClassifier
    """
    print(f"\n[TREE] Training Decision Tree (max_depth={max_depth})...")
    
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=ExplainConfig.TREE_MIN_SAMPLES_SPLIT,
        min_samples_leaf=ExplainConfig.TREE_MIN_SAMPLES_LEAF,
        random_state=42,
        class_weight='balanced',  # Handle action imbalance
    )
    
    dt_classifier.fit(X, y)
    
    print(f"[TREE] Tree trained successfully!")
    print(f"[TREE] Tree depth: {dt_classifier.get_depth()}")
    print(f"[TREE] Number of leaves: {dt_classifier.get_n_leaves()}")
    
    return dt_classifier


# --- Fidelity Score Calculation ---
def calculate_fidelity(agent, dt_classifier, X, y, env=None):
    """
    Calculate how well the Decision Tree mimics the PPO agent.
    
    Fidelity is measured as:
    - Tree accuracy on the collected dataset (ideally should be high)
    - Agreement between tree predictions and agent predictions on new data
    
    Args:
        agent: Trained PPO agent
        dt_classifier: Trained Decision Tree
        X: Observations used for training the tree
        y: Ground truth actions from agent
        env: SumoEnv instance (optional, for fresh validation samples)
    
    Returns:
        fidelity_metrics: Dictionary with fidelity scores
    """
    print(f"\n[FIDELITY] Calculating fidelity scores...")
    
    # Score 1: Tree accuracy on training data
    tree_pred = dt_classifier.predict(X)
    tree_accuracy = accuracy_score(y, tree_pred)
    
    print(f"[FIDELITY] Tree accuracy on collected data: {tree_accuracy:.4f}")
    
    # Score 2: Agreement on same data
    agreement = np.mean(tree_pred == y)
    print(f"[FIDELITY] Agreement between Tree and Agent: {agreement:.4f}")
    
    # Score 3: Feature importance
    feature_importance = dt_classifier.feature_importances_
    important_idx = np.argsort(feature_importance)[::-1][:5]  # Top 5
    
    feature_names = get_feature_names()
    print(f"\n[FIDELITY] Top 5 most important features:")
    for rank, idx in enumerate(important_idx, 1):
        importance = feature_importance[idx]
        print(f"  {rank}. {feature_names[idx]}: {importance:.4f}")
    
    metrics = {
        "tree_accuracy": float(tree_accuracy),
        "agent_agreement": float(agreement),
        "tree_depth": int(dt_classifier.get_depth()),
        "tree_leaves": int(dt_classifier.get_n_leaves()),
        "num_samples": int(len(X)),
        "top_features": [
            {
                "rank": int(rank),
                "feature": feature_names[idx],
                "importance": float(feature_importance[idx])
            }
            for rank, idx in enumerate(important_idx, 1)
        ]
    }
    
    return metrics


# --- Visualization ---
def visualize_tree(dt_classifier, feature_names, output_path, dpi=300):
    """
    Visualize the Decision Tree and save as PNG with enhanced readability.
    
    Args:
        dt_classifier: Trained DecisionTreeClassifier
        feature_names: List of feature names
        output_path: Path to save the PNG file
        dpi: Resolution (dots per inch)
    """
    print(f"\n[VISUALIZATION] Creating tree visualization...")
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(30, 20), dpi=100)
    
    plot_tree(
        dt_classifier,
        feature_names=feature_names,
        class_names=["Keep Phase", "Switch Phase"],
        filled=True,
        rounded=True,
        fontsize=12,
        ax=ax,
        proportion=True,  # Show percentages instead of raw counts
        precision=2,  # 2 decimal places
    )
    
    plt.title(
        "PPO Traffic Signal Policy Logic\n(Decision Tree Distillation)\n"
        "How the agent decides when to KEEP or SWITCH traffic light phases",
        fontsize=20,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
    print(f"[VISUALIZATION] Tree visualization saved to {output_path}")
    plt.close()
    
    # Also create a simplified text-based interpretation
    _save_tree_interpretation(dt_classifier, feature_names, output_path.replace('.png', '_interpretation.txt'))


# --- Tree Interpretation ---
def _save_tree_interpretation(dt_classifier, feature_names, output_path):
    """
    Create a human-readable text explanation of the decision tree.
    
    Args:
        dt_classifier: Trained DecisionTreeClassifier
        feature_names: List of feature names
        output_path: Path to save the interpretation file
    """
    try:
        interpretation_lines = []
        
        # Header
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("TRAFFIC SIGNAL CONTROL DECISION TREE INTERPRETATION")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("")
        interpretation_lines.append("ACTION MEANINGS:")
        interpretation_lines.append("  Class 0: KEEP PHASE - Continue current traffic light phase")
        interpretation_lines.append("  Class 1: SWITCH PHASE - Change to next traffic light phase")
        interpretation_lines.append("")
        
        # Top features
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("TOP 5 DECISION FACTORS (Most Important)")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("")
        
        feature_importance = dt_classifier.feature_importances_
        important_idx = np.argsort(feature_importance)[::-1][:5]
        
        for rank, idx in enumerate(important_idx, 1):
            importance = feature_importance[idx]
            interpretation_lines.append(
                f"{rank}. {feature_names[idx]:25s} - Importance: {importance:.4f} ({100*importance:.1f}%)"
            )
        
        interpretation_lines.append("")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("TREE STATISTICS")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("")
        interpretation_lines.append(f"Tree Depth:        {dt_classifier.get_depth()}")
        interpretation_lines.append(f"Number of Leaves:  {dt_classifier.get_n_leaves()}")
        interpretation_lines.append(f"Tree Accuracy:     {(dt_classifier.score([],[]) if False else 'See tree_summary.json')}")
        
        interpretation_lines.append("")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("HOW TO READ THE VISUALIZATION:")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("")
        interpretation_lines.append("1. START AT THE TOP:")
        interpretation_lines.append("   The top node shows the first decision point (usually Queue_Lane_X)")
        interpretation_lines.append("")
        interpretation_lines.append("2. FOLLOW LEFT/RIGHT BRANCHES:")
        interpretation_lines.append("   - LEFT branch: if feature <= threshold")
        interpretation_lines.append("   - RIGHT branch: if feature > threshold")
        interpretation_lines.append("")
        interpretation_lines.append("3. READ THE NODE INFORMATION:")
        interpretation_lines.append("   - Feature name & threshold value at top")
        interpretation_lines.append("   - Percentages (12% / 88%) show action distribution")
        interpretation_lines.append("   - Color indicates dominant decision")
        interpretation_lines.append("     * Blue = KEEP PHASE (higher percentage)")
        interpretation_lines.append("     * Orange/Red = SWITCH PHASE (higher percentage)")
        interpretation_lines.append("")
        interpretation_lines.append("4. REACH A LEAF (BOTTOM NODE):")
        interpretation_lines.append("   This shows the final decision for that condition")
        interpretation_lines.append("")
        
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("EXAMPLE INTERPRETATION:")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("")
        interpretation_lines.append("IF  Queue_Lane_3 <= 5.0:")
        interpretation_lines.append("  AND Bus_Lane_0 <= 2.0:")
        interpretation_lines.append("    THEN: SWITCH PHASE (because queues are small, room for others)")
        interpretation_lines.append("")
        interpretation_lines.append("ELSE (Queue_Lane_3 > 5.0):")
        interpretation_lines.append("    THEN: KEEP PHASE (because queue is too long, don't switch)")
        interpretation_lines.append("")
        
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("KEY INSIGHTS:")
        interpretation_lines.append("=" * 80)
        interpretation_lines.append("")
        interpretation_lines.append(f"Most Important Feature: {feature_names[important_idx[0]]}")
        interpretation_lines.append("This sensor has the most influence on switching decisions.")
        interpretation_lines.append("")
        interpretation_lines.append("Less Important Features:")
        interpretation_lines.append("These features at the bottom of the tree fine-tune decisions")
        interpretation_lines.append("when the top conditions are ambiguous.")
        interpretation_lines.append("")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(interpretation_lines))
        
        print(f"[VISUALIZATION] Tree interpretation saved to {output_path}")
        print("\n" + "=" * 80)
        print("DECISION TREE EXPLANATION:")
        print("=" * 80)
        for line in interpretation_lines[:30]:
            print(line)
        print("...[see full details in decision_tree_logic_interpretation.txt]...")
        
    except Exception as e:
        print(f"[WARNING] Could not create interpretation: {e}")
        print("[INFO] But the visualization PNG should still be available!")



# --- Main Pipeline ---
def explain_policy(model_path=None, episodes=10, stochastic=False):
    """
    Main function to explain PPO policy using Decision Tree distillation.
    
    Args:
        model_path: Path to trained PPO model (default: from config)
        episodes: Number of episodes for data collection (default: from config)
        stochastic: If True, sample actions probabilistically (more diversity)
    """
    if model_path is None:
        model_path = ExplainConfig.MODEL_PATH
    if episodes is None:
        episodes = ExplainConfig.COLLECTION_EPISODES
    
    print("\n" + "=" * 70)
    print("[POLICY EXPLANATION] Starting Policy Distillation Pipeline")
    print("=" * 70)
    
    # --- 1. Setup ---
    print(f"\n[SETUP] Output directory: {ExplainConfig.OUTPUT_DIR}")
    os.makedirs(ExplainConfig.OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Load Agent ---
    print(f"\n[LOAD] Loading trained PPO agent from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent = load_ppo_agent(model_path)
    print("[LOAD] Agent loaded successfully!")
    
    # --- 3. Create Environment ---
    print(f"\n[ENV] Creating evaluation environment...")
    env = SumoEnv(
        use_gui=ExplainConfig.USE_GUI,
        sumocfg_file=ExplainConfig.SUMOCFG_FILE,
    )
    env.action_space.seed(ExplainConfig.COLLECTION_SEED)
    print("[ENV] Environment created successfully!")
    
    # --- 4. Collect Data ---
    X, y = collect_data(agent, env, episodes=episodes, stochastic=stochastic)
    
    # --- 5. Get Feature Names ---
    feature_names = get_feature_names()
    print(f"\n[FEATURES] Feature space: {len(feature_names)} features")
    
    # --- 6. Check Action Balance ---
    action_counts = np.bincount(y)
    if len(action_counts) == 1:
        print(f"\n[WARNING] Only one action is present in collected data!")
        print(f"[WARNING] The agent exclusively chose action {np.argmax(action_counts)}")
        print(f"[SUGGESTION] Try:")
        print(f"  1. Using --stochastic flag to sample actions probabilistically")
        print(f"  2. Increasing episodes (--episodes 20+)")
        print(f"  3. Using a different model or running during training (not final model)")
    
    # --- 7. Train Decision Tree ---
    dt_classifier = train_decision_tree(X, y, feature_names, max_depth=ExplainConfig.TREE_MAX_DEPTH)
    
    # --- 8. Calculate Fidelity ---
    fidelity_metrics = calculate_fidelity(agent, dt_classifier, X, y, env)
    
    # --- 9. Visualize Tree ---
    tree_png_path = os.path.join(ExplainConfig.OUTPUT_DIR, ExplainConfig.TREE_PNG)
    visualize_tree(dt_classifier, feature_names, tree_png_path)
    
    # --- 10. Save Summary ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "episodes_collected": episodes,
        "sampling_mode": "stochastic" if stochastic else "deterministic",
        "fidelity_metrics": fidelity_metrics,
        "feature_names": feature_names,
        "tree_params": {
            "max_depth": ExplainConfig.TREE_MAX_DEPTH,
            "min_samples_split": ExplainConfig.TREE_MIN_SAMPLES_SPLIT,
            "min_samples_leaf": ExplainConfig.TREE_MIN_SAMPLES_LEAF,
        }
    }
    
    summary_path = os.path.join(ExplainConfig.OUTPUT_DIR, ExplainConfig.TREE_SUMMARY)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Summary saved to {summary_path}")
    
    # --- 11. Print Results ---
    print("\n" + "=" * 70)
    print("[RESULTS] Policy Distillation Summary")
    print("=" * 70)
    print(f"Tree Accuracy (on collected data): {fidelity_metrics['tree_accuracy']:.4f}")
    print(f"Agent-Tree Agreement: {fidelity_metrics['agent_agreement']:.4f}")
    print(f"Tree Depth: {fidelity_metrics['tree_depth']}")
    print(f"Tree Leaves: {fidelity_metrics['tree_leaves']}")
    print(f"Total Samples: {fidelity_metrics['num_samples']}")
    print(f"\nVisualization saved: {tree_png_path}")
    print(f"Summary saved: {summary_path}")
    print("=" * 70 + "\n")
    
    # Cleanup
    env.close()
    
    return dt_classifier, fidelity_metrics, feature_names


# --- Entry Point ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Explain PPO agent policy using Decision Tree distillation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ppo_mg_road/best_model.zip",
        help="Path to trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to collect data from",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum depth of decision tree",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions probabilistically (more diverse) instead of deterministically",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable SUMO GUI during data collection",
    )
    
    args = parser.parse_args()
    
    # Update config from CLI arguments
    ExplainConfig.TREE_MAX_DEPTH = args.depth
    ExplainConfig.USE_GUI = args.gui
    
    try:
        dt_classifier, metrics, features = explain_policy(
            model_path=args.model,
            episodes=args.episodes,
            stochastic=args.stochastic,
        )
        print("\n[SUCCESS] Policy explanation completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Policy explanation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
