"""Training entry point for minimal hand latent project."""

from __future__ import annotations

import argparse
import os
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HandLatent.model import CrossEmbodimentTrainer, TrainingConfig


def main() -> None:
    """Run latent training with default parameters matching the reference repo.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Trains the model and writes checkpoints.
    """

    parser = argparse.ArgumentParser(
        description="Train minimal cross-embodiment hand latent model."
    )
    parser.add_argument("--num_steps", type=int, default=15_000)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--pinch_template_count", type=int, default=2048)
    parser.add_argument("--pinch_template_iterations", type=int, default=100)
    parser.add_argument("--lambda_sem_dis", type=float, default=1000.0)
    parser.add_argument("--lambda_sem_dir", type=float, default=50.0)
    parser.add_argument("--lambda_dir", type=float, default=50.0)
    parser.add_argument("--lambda_kl", type=float, default=0.0001)
    parser.add_argument("--pinch_sampling_probability", type=float, default=0.6)
    parser.add_argument("--semantic_weight_floor", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    hand_names = [
        # ── 五指手 (right) ───────────────────────────────────────────
        "xarm7_xhand_right",
        "xarm7_ability_right",
        "xarm7_inspire_right",
        "xarm7_paxini_right",
        # "xarm7_shadow_right",
        # ── 四指手 (right) ───────────────────────────────────────────
        "xarm7_allegro_right",
        "xarm7_leap_right",
        # ── 三指手 (right) ───────────────────────────────────────────
        "xarm7_unitree_right",
        # ── 夹爪 symmetric (right 安装侧) ────────────────────────────
        "xarm7_dclaw_right",
        "xarm7_panda_gripper_right",
        "xarm7_umi_gripper_right",
        # ── 双臂左侧：按需解注释 ─────────────────────────────────────
        # "xarm7_xhand_left",
        # "xarm7_ability_left",
        # "xarm7_inspire_left",
        # "xarm7_paxini_left",
        # "xarm7_shadow_left",
        # "xarm7_allegro_left",
        # "xarm7_leap_left",
        # "xarm7_unitree_left",
        # "xarm7_dclaw_left",
        # "xarm7_panda_gripper_left",
        # "xarm7_umi_gripper_left",
    ]
    config = TrainingConfig(
        num_steps=args.num_steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_min=args.lr_min,
        pinch_template_count=args.pinch_template_count,
        pinch_template_iterations=args.pinch_template_iterations,
        lambda_sem_dis=args.lambda_sem_dis,
        lambda_sem_dir=args.lambda_sem_dir,
        lambda_dir=args.lambda_dir,
        lambda_kl=args.lambda_kl,
        pinch_sampling_probability=args.pinch_sampling_probability,
        semantic_weight_floor=args.semantic_weight_floor,
    )
    trainer = CrossEmbodimentTrainer(hand_names, config)
    trainer.train()


if __name__ == "__main__":
    main()
