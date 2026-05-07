"""Training entry point for minimal hand latent project."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

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
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True, help="Enable SwanLab metric logging.")
    parser.add_argument("--swanlab_project", type=str, default="DexLatent")
    parser.add_argument("--swanlab_workspace", type=str, default=None)
    parser.add_argument("--swanlab_experiment", type=str, default=None)
    parser.add_argument("--swanlab_mode", type=str, default=None, choices=("cloud", "local", "offline", "disabled"))
    parser.add_argument("--swanlab_logdir", type=str, default="swanlog")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))
        print("allocated MB:", torch.cuda.memory_allocated(0) / 1024 / 1024)
    else:
        print("using CPU")

    hand_names = [
        # ── 五指手 (right) ───────────────────────────────────────────
        "xarm7_xhand1pro_right",
        "xarm7_xhand_right",
        # "xarm7_ability_right",
        "xarm7_inspire_right",
        # "xarm7_paxini_right",
        # "xarm7_shadow_right",
        # ── 四指手 (right) ───────────────────────────────────────────
        # "xarm7_allegro_right",
        # "xarm7_leap_right",
        # ── 三指手 (right) ───────────────────────────────────────────
        "xarm7_unitree_right",
        # ── 夹爪 symmetric (right 安装侧) ────────────────────────────
        # "xarm7_dclaw_right", # 构型相差太大，难以训练
        "xarm7_panda_gripper_right",
        # "xarm7_umi_gripper_right",
        # ── 双臂左侧：按需解注释 ─────────────────────────────────────
        # "xarm7_xhand1pro_left",
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

    swanlab_run = None
    log_callback = None
    if args.swanlab:
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging requested but swanlab is not installed. "
                "Install it or run without --swanlab."
            ) from exc

        experiment_name = args.swanlab_experiment or f"train_xeef_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        swanlab_config: Dict[str, Any] = {
            "script": "HandLatent/train_xeef.py",
            "hand_names": hand_names,
            "seed": args.seed,
            "training_config": {
                key: str(value) if isinstance(value, torch.device) else value
                for key, value in asdict(config).items()
            },
        }
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            experiment_name=experiment_name,
            config=swanlab_config,
            logdir=args.swanlab_logdir,
            mode=args.swanlab_mode,
        )

        def log_callback(step: int, metrics: Dict[str, float]) -> None:
            swanlab.log(metrics, step=step)

    try:
        trainer.train(log_callback=log_callback)
    finally:
        if swanlab_run is not None:
            swanlab.finish()


if __name__ == "__main__":
    main()
