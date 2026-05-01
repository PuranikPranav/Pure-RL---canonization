import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Ensure project root is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoImageProcessor, AutoModelForImageClassification
from src.utils import load_config, get_device
from src.dataset import load_pool_from_dir
from src.ppo import load_policy_state
from src.policy import CanonicalizationPolicy, ImageEncoderPreprocessor
from src.env import ActionSpace
from src.rotation import rotate_image
from src.reward_model import build_reward_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Force device to 'mps' if available for M4 Pro
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[eval] Using device: {device}")

    # 1. Load Classifier (Downstream CNN)
    model_name = "facebook/dinov2-small-imagenet1k-1-layer"
    print(f"[eval] Loading downstream classifier: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    classifier = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    classifier.eval()

    # 2. Load Dataset
    print(f"[eval] Loading images from {cfg['data']['dir']}")
    pool = load_pool_from_dir(cfg["data"]["dir"], image_size=224, max_images=args.num_images)
    
    # 3. Load your PPO Policy
    action_space = ActionSpace(bound=cfg["action"]["bound"], step_size=cfg["action"]["step_size"])
    policy = CanonicalizationPolicy(
        backbone_name=cfg["policy"]["backbone"],
        num_actions=action_space.n,
        hidden_dim=cfg["policy"]["hidden_dim"]
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_policy_state(policy, ckpt)
    policy.eval()

    preprocessor = ImageEncoderPreprocessor(cfg["policy"]["backbone"])

    print("\n--- Starting Detailed Impact Evaluation ---")
    
    results_rotated = []
    results_fixed = []
    
    # Create a directory to store the results so they don't clutter your main folder
    output_path = Path("experiment_results_radios")
    output_path.mkdir(exist_ok=True)

    for i in range(len(pool.images)):
        orig_img = pool.get(i)
        
        # A. Get Ground Truth Label
        pil_orig = Image.fromarray(orig_img).convert("RGB")
        inputs_orig = processor(images=pil_orig, return_tensors="pt").to(device)
        with torch.no_grad():
            label = classifier(**inputs_orig).logits.argmax(-1).item()
        
        # B. Apply Random Rotation
        initial_angle = np.random.uniform(-90, 90)
        current_img = rotate_image(orig_img, initial_angle)
        
        # C. Test Baseline Accuracy (Rotated)
        pil_rot = Image.fromarray(current_img.astype(np.uint8)).convert("RGB")
        inputs_rot = processor(images=pil_rot, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_rot = classifier(**inputs_rot).logits.argmax(-1).item()
        results_rotated.append(pred_rot == label)
        
        # D. PPO Fix Loop + Track Rotational Impact
        angle_history = [initial_angle]
        current_angle = initial_angle
        
        for step in range(30): # 30 steps is enough to see a trend
            obs_tensor = preprocessor(current_img[None, ...]).to(device)
            with torch.no_grad():
                action_idx, _, _ = policy.act(obs_tensor, greedy=True)
            
            # Map index to degrees (assuming 11 actions, -5 to +5)
            delta = (action_idx.item() - (action_space.n // 2)) * cfg["action"]["step_size"]
            
            if abs(delta) < 0.1: # Agent decided to stop
                break
                
            current_angle += delta
            angle_history.append(current_angle)
            current_img = rotate_image(orig_img, current_angle)

        # E. Test PPO Accuracy (Fixed)
        pil_fixed = Image.fromarray(current_img.astype(np.uint8)).convert("RGB")
        inputs_fixed = processor(images=pil_fixed, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_fixed = classifier(**inputs_fixed).logits.argmax(-1).item()
        results_fixed.append(pred_fixed == label)

        # F. SAVE THE VISUAL IMPACT (Every 5th image to save space)
        if i % 5 == 0:
            comparison = Image.new('RGB', (448, 224))
            comparison.paste(pil_rot, (0, 0))   # Left: Rotated
            comparison.paste(pil_fixed, (224, 0)) # Right: PPO Fixed
            comparison.save(output_path / f"comparison_sample_{i}.png")
            print(f"Sample {i}: Initial Angle {initial_angle:.1f} -> Final {current_angle:.1f}. History: {angle_history}")

    # (Keep your final print summary at the bottom)

    # Final Summary
    acc_rot = sum(results_rotated) / len(results_rotated)
    acc_fixed = sum(results_fixed) / len(results_fixed)

    print("\n" + "="*40)
    print("FINAL ACCURACY COMPARISON")
    print("="*40)
    print(f"Baseline Accuracy (Rotated):  {acc_rot * 100:.2f}%")
    print(f"PPO Accuracy (After Fix):     {acc_fixed * 100:.2f}%")
    print("-" * 40)
    print(f"Net Project Gain: {(acc_fixed - acc_rot)*100:+.2f} points")
    print("="*40)

    

if __name__ == "__main__":
    main()