"""SAPIEN-backend inference CLI — parallel to infer_xeef.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from HandLatent.infer_xeef import (
    EvaluationConfig,
    encode_hand_sequence_eepose,
    decode_hand_sequence_eepose,
    _find_latest_checkpoint,
)
from HandLatent.model import (
    CrossEmbodimentTrainer,
    TrainingConfig,
)
from HandLatent.sapien_visualize import (
    create_sapien_scene,
    load_hand_trajectory,
    run_sapien_replay,
)


# Horizontal spacing between robots in the X direction (metres)
_X_SPACING: float = 1.0
# Y offset separating left-side robots from right-side (metres)
_Y_SIDE_OFFSET: float = 0.8


def main() -> None:
    """Run inference and open a SAPIEN viewer for interactive replay."""

    parser = argparse.ArgumentParser(
        description="Retarget one real trajectory and visualize in SAPIEN."
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--data", type=str, default="Dataset/episode_38.npz")
    parser.add_argument(
        "--side", type=str, default="both", choices=("right", "left", "both")
    )
    parser.add_argument(
        "--frame-sleep",
        type=float,
        default=0.02,
        help="Seconds per frame during auto-play (default: 0.02).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable looping; stop at the last frame.",
    )
    args = parser.parse_args()

    sides = ("right", "left") if args.side == "both" else (args.side,)
    trainer_hands = [
        hand_name
        for side in sides
        for hand_name in (
            f"xarm7_xhand_{side}",
            f"xarm7_inspire_{side}",
            f"xarm7_unitree_{side}",
            f"xarm7_panda_gripper_{side}",
        )
    ]

    config = TrainingConfig()
    trainer = CrossEmbodimentTrainer(trainer_hands, config)
    ckpt_path = (
        Path(args.ckpt).expanduser().resolve()
        if args.ckpt is not None
        else _find_latest_checkpoint(Path(config.checkpoint_dir))
    )
    payload = torch.load(ckpt_path, map_location="cpu")
    trainer.load_autoencoders_from_payload(payload)

    scene = create_sapien_scene()
    all_instances = []

    with np.load(Path(args.data).expanduser().resolve()) as dataset:
        for side in sides:
            source_hand = f"xarm7_inspire_{side}"
            target_hands = [
                f"xarm7_xhand_{side}",
                f"xarm7_inspire_{side}",
                f"xarm7_unitree_{side}",
                f"xarm7_panda_gripper_{side}",
            ]

            source_qpos = torch.as_tensor(
                dataset[f"{side}_qpos"], dtype=torch.float32
            )
            source_norm = trainer.normalized_qpos(source_hand, source_qpos).to(
                device=config.device
            )

            with torch.no_grad():
                latents = encode_hand_sequence_eepose(
                    trainer=trainer,
                    hand_name=source_hand,
                    qpos=source_norm,
                )
                decoded: dict = {
                    hand_name: decode_hand_sequence_eepose(
                        trainer=trainer,
                        hand_name=hand_name,
                        latents=latents,
                        evaluation_config=EvaluationConfig(),
                    )
                    .detach()
                    .cpu()
                    for hand_name in target_hands
                }

            y_offset = 0.0 if side == "right" else _Y_SIDE_OFFSET
            T = source_norm.shape[0]

            # Per-frame root offsets: all frames share the same XY plane
            # position; the robots stay at their initial base_pose.  We pass
            # None here because spacing is encoded in base_pose directly.
            # (Kept as np.zeros for API symmetry; replace with non-zero arrays
            # if dynamic root motion is needed.)
            static_offsets = np.zeros((T, 3), dtype=np.float32)

            # Source robot at X = 0
            source_array = source_norm.detach().cpu().numpy()
            source_instance = load_hand_trajectory(
                hand_name=source_hand,
                joint_series=source_array,
                scene=scene,
                base_pose=np.array([0.0, y_offset, 0.0], dtype=np.float32),
                per_frame_root_offsets=static_offsets,
                label=f"{source_hand}_origin_{side}",
            )
            all_instances.append(source_instance)

            # Decoded robots at X = 1, 2, …
            for col_idx, hand_name in enumerate(target_hands, start=1):
                series = decoded[hand_name].numpy()
                inst = load_hand_trajectory(
                    hand_name=hand_name,
                    joint_series=series,
                    scene=scene,
                    base_pose=np.array(
                        [col_idx * _X_SPACING, y_offset, 0.0], dtype=np.float32
                    ),
                    per_frame_root_offsets=static_offsets,
                    label=f"{hand_name}_decode_{side}",
                )
                all_instances.append(inst)

    # Centre camera on the mid-point of all robots
    n_cols = 1 + len(target_hands)
    cam_x = (n_cols - 1) * _X_SPACING * 0.5
    cam_y_right = -2.0
    run_sapien_replay(
        instances=all_instances,
        scene=scene,
        frame_sleep=args.frame_sleep,
        loop=not args.no_loop,
        camera_xyz=(cam_x, cam_y_right, 1.2),
        camera_rpy=(0.0, -0.35, np.pi / 2),
        window_title="HandLatent SAPIEN — xEEF Retargeting",
    )


if __name__ == "__main__":
    main()
