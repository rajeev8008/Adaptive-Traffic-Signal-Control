"""
SHAP Feature Attribution Analysis
===================================
Uses SHAP (SHapley Additive exPlanations) to explain why the PPO agent
makes specific decisions, particularly when emergency vehicles appear.

This provides a local explanations (force plots) and global importance (summary plots)
showing which features (queue lengths, emergency counts, etc.) drive the agent's decisions.

Usage:
    python explain_shap.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shap
import torch
from datetime import datetime
from typing import Callable, Tuple, List

# Add parent directory to path for imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.SumoEnv import SumoEnv, INCOMING_LANES
from utils.ppo_agent import load_ppo_agent


def get_feature_names(num_lanes: int) -> List[str]:
    """
    Generate feature names matching the observation structure.
    
    Observation structure (3*num_lanes + 1):
    - [0:num_lanes]: Queue lengths
    - [num_lanes]: Phase indicator
    - [num_lanes+1:2*num_lanes+1]: Emergency counts
    - [2*num_lanes+1:3*num_lanes+1]: Bus counts
    
    Args:
        num_lanes: Number of incoming lanes
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # Queue lengths
    for i in range(num_lanes):
        feature_names.append(f"Queue_Lane_{i}")
    
    # Phase indicator
    feature_names.append("Phase")
    
    # Emergency counts
    for i in range(num_lanes):
        feature_names.append(f"Emerg_Lane_{i}")
    
    # Bus counts
    for i in range(num_lanes):
        feature_names.append(f"Bus_Lane_{i}")
    
    return feature_names


def create_prediction_function(agent, device: str = "cpu") -> Callable:
    """
    Create a wrapper function for SHAP that predicts the probability of action 1 (Switch).
    
    SHAP expects a function f(X) where:
    - X: numpy array of shape (num_samples, num_features)
    - Returns: numpy array of shape (num_samples,) with action probabilities
    
    Args:
        agent: Trained PPO agent
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Prediction function compatible with SHAP
    """
    def predict_proba(obs_array: np.ndarray) -> np.ndarray:
        """
        Predict probability of Action 1 (Switch Phase) for batch of observations.
        
        Args:
            obs_array: Shape (num_samples, num_features)
            
        Returns:
            Probabilities of action 1, shape (num_samples,)
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Get policy distribution
        distribution = agent.policy.get_distribution(obs_tensor)
        
        # Get action probabilities
        probs = distribution.distribution.probs.detach().cpu().numpy()
        
        # Return probability of action 1 (Switch)
        return probs[:, 1]
    
    return predict_proba


def collect_background_data(
    agent,
    env: SumoEnv,
    num_steps: int = 100,
) -> np.ndarray:
    """
    Collect representative background observations for SHAP.
    
    Args:
        agent: Trained PPO agent
        env: SUMO environment
        num_steps: Number of steps to collect
        
    Returns:
        Array of observations, shape (num_steps, num_features)
    """
    print(f"\n[BACKGROUND] Collecting {num_steps} representative observations...")
    
    obs, _ = env.reset()
    background_data = []
    steps = 0
    done = False
    
    while steps < num_steps and not done:
        background_data.append(obs.copy())
        
        # Use agent to step (deterministic policy)
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        steps += 1
        if (steps + 1) % 25 == 0:
            print(f"  Collected {steps}/{num_steps} observations")
    
    print(f"[BACKGROUND] Collected {len(background_data)} observations")
    return np.array(background_data, dtype=np.float32)


def find_emergency_observation(
    agent,
    env: SumoEnv,
    num_lanes: int,
    max_attempts: int = 10,
    max_steps_per_attempt: int = 500,
) -> Tuple[np.ndarray, bool]:
    """
    Run simulation until an emergency vehicle appears and capture that observation.
    
    Emergency vehicles are indicated by obs[num_lanes+1:2*num_lanes+1] > 0
    
    Args:
        agent: Trained PPO agent
        env: SUMO environment
        num_lanes: Number of incoming lanes
        max_attempts: Maximum episodes to try
        max_steps_per_attempt: Maximum steps per episode
        
    Returns:
        Tuple of (observation_with_emergency, found_emergency)
    """
    print(f"\n[EMERGENCY] Looking for emergency vehicle in observation...")
    
    for attempt in range(max_attempts):
        print(f"  Attempt {attempt + 1}/{max_attempts}")
        obs, _ = env.reset()
        
        # Recalculate actual number of lanes from observation size
        obs_size = len(obs)
        actual_num_lanes = (obs_size - 1) // 3
        
        steps = 0
        done = False
        
        while steps < max_steps_per_attempt and not done:
            # Check if emergency vehicle is present
            # Emergency counts are at indices [actual_num_lanes+1 : 2*actual_num_lanes+1]
            emergency_lane_start = actual_num_lanes + 1
            emergency_lane_end = 2 * actual_num_lanes + 1
            
            if emergency_lane_end <= len(obs):
                emergency_counts = obs[emergency_lane_start:emergency_lane_end]
                
                if np.any(emergency_counts > 0):
                    print(f"  ✓ Found emergency vehicle at step {steps}!")
                    print(f"    Emergency counts: {emergency_counts[emergency_counts > 0]}")
                    return obs.copy(), True
            
            # Step the environment
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        print(f"    No emergency found in this attempt (ran {steps} steps)")
    
    # If no emergency found, return last observation
    print(f"[WARNING] No emergency vehicle found after {max_attempts} attempts")
    return obs.copy(), False


def create_shap_explainer(
    background_data: np.ndarray,
    predict_func: Callable,
    num_features: int,
) -> shap.KernelExplainer:
    """
    Create a SHAP KernelExplainer.
    
    Args:
        background_data: Background dataset
        predict_func: Prediction function
        num_features: Number of features
        
    Returns:
        SHAP KernelExplainer instance
    """
    print(f"\n[SHAP] Initializing KernelExplainer...")
    print(f"  Background data shape: {background_data.shape}")
    print(f"  Number of features: {num_features}")
    
    explainer = shap.KernelExplainer(
        model=predict_func,
        data=background_data,
        feature_names=[f"Feature_{i}" for i in range(num_features)],  # Will be updated
        link="identity"
    )
    
    return explainer


def plot_summary_plot(
    explainer: shap.KernelExplainer,
    background_data: np.ndarray,
    feature_names: List[str],
    output_path: str = "shap_summary_plot.png"
):
    """
    Generate and save SHAP summary plot (bar plot of mean absolute SHAP values).
    
    Args:
        explainer: SHAP explainer
        background_data: Background observations
        feature_names: Feature names
        output_path: Path to save plot
    """
    print(f"\n[SUMMARY PLOT] Computing SHAP values for background data...")
    
    # Compute SHAP values for background data
    shap_values = explainer.shap_values(background_data)
    
    # Handle case where shap_values is a list (for binary classification)
    if isinstance(shap_values, list):
        # Take action 1 (Switch)
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]
    
    # Ensure shap_values is 2D (num_samples, num_features)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)
    
    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    feature_importance_indices = np.argsort(mean_abs_shap)[::-1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot top features
    top_n = min(20, len(feature_names))
    top_indices = feature_importance_indices[:top_n]
    
    y_pos = np.arange(len(top_indices))
    values = mean_abs_shap[top_indices]
    labels = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices]
    
    ax.barh(y_pos, values, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Summary Plot: Global Feature Importance\n(Why Agent Switches Traffic Phase)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SUMMARY PLOT] Saved to {output_path}")
    plt.close()
    
    return shap_values


def plot_force_plot(
    explainer: shap.KernelExplainer,
    obs_emergency: np.ndarray,
    feature_names: List[str],
    output_path: str = "shap_force_plot.html"
):
    """
    Generate and save SHAP force plot for emergency observation.
    
    Args:
        explainer: SHAP explainer
        obs_emergency: Observation with emergency vehicle
        feature_names: Feature names
        output_path: Path to save plot
    """
    print(f"\n[FORCE PLOT] Computing SHAP values for emergency observation...")
    
    # Reshape observation for prediction
    obs_emergency_batch = obs_emergency.reshape(1, -1)
    
    # Compute SHAP values
    shap_value = explainer.shap_values(obs_emergency_batch)[0]
    
    # Get base value (expected model output)
    base_value = explainer.expected_value
    
    # Create force plot
    force_plot = shap.force_plot(
        base_value,
        shap_value,
        obs_emergency,
        feature_names=feature_names,
        show=False,
    )
    
    # Save force plot
    shap.save_html(output_path, force_plot)
    print(f"[FORCE PLOT] Saved to {output_path}")
    
    return shap_value


def print_feature_importance(
    shap_value: np.ndarray,
    feature_names: List[str],
    obs: np.ndarray,
    top_k: int = 10
):
    """
    Print most important features for a decision.
    
    Args:
        shap_value: SHAP values
        feature_names: Feature names
        obs: Observation values
        top_k: Number of top features to print
    """
    print(f"\n[IMPORTANCE] Top {top_k} features pushing agent to SWITCH phase:")
    print("=" * 80)
    
    # Ensure shap_value is 1D
    if isinstance(shap_value, np.ndarray):
        if shap_value.ndim > 1:
            shap_value = shap_value.flatten()
    
    # Validate that we have matching dimensions
    if len(shap_value) != len(obs):
        print(f"[WARNING] SHAP values ({len(shap_value)}) don't match observation ({len(obs)})")
        print(f"          Limiting to available features")
        top_k = min(top_k, len(shap_value), len(feature_names))
    else:
        top_k = min(top_k, len(shap_value))
    
    # Get absolute SHAP values to identify importance magnitude
    importance_indices = np.argsort(np.abs(shap_value[:len(feature_names)]))[::-1][:top_k]
    
    # Print in order of importance
    for rank, idx in enumerate(importance_indices, 1):
        if idx >= len(feature_names):
            continue
        feature_name = feature_names[idx]
        shap_val = shap_value[idx] if idx < len(shap_value) else 0.0
        obs_val = obs[idx] if idx < len(obs) else 0.0
        direction = "↑ PUSH TO SWITCH" if shap_val > 0 else "↓ PUSH TO HOLD"
        
        print(f"{rank:2d}. {feature_name:20s} | Value: {obs_val:8.2f} | SHAP: {shap_val:+.4f} | {direction}")
    
    print("=" * 80)


def main():
    """Main SHAP explanation pipeline."""
    print("\n" + "=" * 80)
    print("SHAP FEATURE ATTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # === Setup ===
    print("\n[SETUP] Loading environment and agent...")
    try:
        env = SumoEnv(sumocfg_file="SUMO_Trinity_Traffic_sim/osm.sumocfg", use_gui=False)
        print(f"[SETUP] Environment loaded successfully")
        print(f"        Observation space: {env.observation_space.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load environment: {e}")
        return
    
    model_path = "./models/ppo_mg_road/best_model.zip"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        print(f"        Please train the model first: python scripts/train_ppo.py")
        return
    
    try:
        agent = load_ppo_agent(model_path=model_path)
        print(f"[SETUP] Agent loaded from: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load agent: {e}")
        return
    
    # Get number of lanes (may be detected after environment initialization)
    num_features = env.observation_space.shape[0]
    # Infer num_lanes from observation space: obs = queues + phase + emergency + bus
    # obs_size = 3*num_lanes + 1, so num_lanes = (obs_size - 1) / 3
    num_lanes = (num_features - 1) // 3
    print(f"[SETUP] Number of features: {num_features}")
    print(f"[SETUP] Inferred number of lanes: {num_lanes}")
    
    # === Create feature names ===
    feature_names = get_feature_names(num_lanes)
    print(f"[SETUP] Generated {len(feature_names)} feature names")
    
    # === Create output directory ===
    output_dir = Path("./logs/policy_explanation")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Output directory: {output_dir}")
    
    # === Create prediction function ===
    print("\n[PREDICT] Creating prediction wrapper...")
    predict_func = create_prediction_function(agent, device="cpu")
    print(f"[PREDICT] Prediction function created")
    
    # === Collect background data ===
    background_data = collect_background_data(agent, env, num_steps=100)
    
    # === Update feature names based on ACTUAL observation size ===
    actual_num_features = background_data.shape[1]
    actual_num_lanes = (actual_num_features - 1) // 3
    print(f"\n[SETUP] Actual feature count from background data: {actual_num_features}")
    print(f"[SETUP] Actual number of detected lanes: {actual_num_lanes}")
    
    # Regenerate feature names with actual lane count
    feature_names = get_feature_names(actual_num_lanes)
    print(f"[SETUP] Regenerated {len(feature_names)} feature names")
    
    # === Create SHAP explainer ===
    explainer = create_shap_explainer(background_data, predict_func, actual_num_features)
    
    # Update feature names in explainer
    explainer.feature_names = feature_names
    
    # === Find emergency observation ===
    obs_emergency, found_emergency = find_emergency_observation(
        agent, env, actual_num_lanes, max_attempts=10
    )
    
    if not found_emergency:
        print(f"\n[WARNING] Emergency observation not found.")
        print(f"          Using last collected observation instead")
        obs_emergency = background_data[-1]
    
    # === Generate summary plot ===
    print("\n[VISUALIZATION] Generating summary plot...")
    try:
        summary_output = str(output_dir / "shap_summary_plot.png")
        shap_values_background = plot_summary_plot(
            explainer, 
            background_data, 
            feature_names,
            output_path=summary_output
        )
        print(f"✓ Summary plot saved")
    except Exception as e:
        print(f"[WARNING] Summary plot generation failed: {e}")
        print(f"          Continuing with force plot only...")
        shap_values_background = None
    
    # === Generate force plot for emergency ===
    print("\n[VISUALIZATION] Generating force plot for emergency observation...")
    try:
        force_output = str(output_dir / "shap_force_plot.html")
        shap_value_emergency = plot_force_plot(
            explainer,
            obs_emergency,
            feature_names,
            output_path=force_output
        )
        print(f"✓ Force plot saved (open in web browser)")
    except Exception as e:
        print(f"[ERROR] Failed to generate force plot: {e}")
        shap_value_emergency = None
    
    # === Print feature importance ===
    if shap_value_emergency is not None:
        print_feature_importance(
            shap_value_emergency,
            feature_names,
            obs_emergency,
            top_k=10
        )
    
    # === Analysis summary ===
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - shap_summary_plot.png    : Global feature importance (beeswarm)")
    print(f"  - shap_force_plot.html     : Local explanation for emergency event")
    print(f"\nKey Insights:")
    print(f"  • Summary plot shows which features affect decisions globally")
    print(f"  • Force plot explains the specific emergency observation")
    print(f"  • Red features push toward SWITCH action")
    print(f"  • Blue features push toward HOLD action")
    
    if found_emergency and shap_value_emergency is not None:
        top_emergency_features = np.argsort(
            np.abs(shap_value_emergency)
        )[::-1][:3]
        print(f"\nTop 3 features driving emergency response:")
        for idx in top_emergency_features:
            print(f"  • {feature_names[idx]}: value={obs_emergency[idx]:.2f}")
    
    print("\n" + "=" * 80)
    
    env.close()


if __name__ == "__main__":
    main()
