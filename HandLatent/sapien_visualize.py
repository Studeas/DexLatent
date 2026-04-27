"""SAPIEN-based visualization for normalized xarm hand trajectories.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import sapien

from HandLatent.kinematics import HAND_CONFIGS, HandKinematicsModel, load_urdf_silent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_SLEEP_SECONDS: float = 0.02

# Viewer background
_AMBIENT_LIGHT: List[float] = [0.4, 0.4, 0.4]
_DIRECTIONAL_LIGHT_DIR: List[float] = [0.5, -1.0, -1.0]
_DIRECTIONAL_LIGHT_COLOR: List[float] = [0.6, 0.6, 0.6]

# Default camera placement (can be overridden in run_sapien_replay)
_DEFAULT_CAM_XYZ: Tuple[float, float, float] = (0.0, -1.8, 0.8)
_DEFAULT_CAM_RPY: Tuple[float, float, float] = (0.0, -0.35, np.pi / 2)


# ---------------------------------------------------------------------------
# Scene factory
# ---------------------------------------------------------------------------


def create_sapien_scene() -> sapien.Scene:
    """Create and configure a SAPIEN scene ready for kinematic replay.

    Returns
    -------
    sapien.Scene
        Configured scene with ambient and directional lighting.
    """

    scene = sapien.Scene()
    scene.set_ambient_light(_AMBIENT_LIGHT)
    scene.add_directional_light(_DIRECTIONAL_LIGHT_DIR, _DIRECTIONAL_LIGHT_COLOR)
    return scene


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class SapienHandInstance:
    """One loaded articulation with its pre-computed qpos trajectory.

    Parameters
    ----------
    robot : sapien.Articulation
        SAPIEN articulation loaded from the hand URDF.
    trajectory : np.ndarray, shape=(T, D_sapien), dtype=float32
        Denormalized joint positions in SAPIEN's active-joint order.
    base_pose : np.ndarray, shape=(3,), dtype=float32
        Fixed world XYZ origin of this robot (before any per-frame offset).
    per_frame_root_offsets : np.ndarray or None, shape=(T, 3), dtype=float32
        Per-frame translation added on top of ``base_pose``.
    label : str
        Human-readable label used for display.
    """

    robot: sapien.Articulation
    trajectory: np.ndarray
    base_pose: np.ndarray
    per_frame_root_offsets: Optional[np.ndarray]
    label: str

    @property
    def num_frames(self) -> int:
        """Return trajectory length."""
        return len(self.trajectory)

    def apply_frame(self, frame_index: int) -> None:
        """Set articulation state to one trajectory frame.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index; clamped to ``[0, T-1]``.
        """

        t = int(np.clip(frame_index, 0, self.num_frames - 1))
        self.robot.set_qpos(self.trajectory[t])

        if self.per_frame_root_offsets is not None:
            offset_t = int(np.clip(t, 0, len(self.per_frame_root_offsets) - 1))
            pos = self.base_pose + self.per_frame_root_offsets[offset_t].astype(np.float32)
        else:
            pos = self.base_pose

        self.robot.set_root_pose(sapien.Pose(p=pos.tolist()))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _denormalize_qpos(
    joint_series: np.ndarray,
    model: HandKinematicsModel,
) -> np.ndarray:
    """Convert normalized trajectory to physical joint angles/displacements.

    Uses the same ``[-1, 1]`` → ``[lower, upper]`` mapping as
    ``HandKinematicsModel._normalized_to_all_joint_angles``.

    Parameters
    ----------
    joint_series : np.ndarray, shape=(T, D), dtype=float32
        Normalized trajectory in ``model.dof_joints`` order.
    model : HandKinematicsModel
        Provides ``_lower`` / ``_upper`` limits.

    Returns
    -------
    np.ndarray, shape=(T, D), dtype=float32
        Physical joint values clamped to ``[lower, upper]``.
    """

    lower = model._lower.numpy().astype(np.float32)
    upper = model._upper.numpy().astype(np.float32)
    clipped = np.clip(joint_series.astype(np.float32), -1.0, 1.0)
    angles = (clipped + 1.0) * 0.5 * (upper - lower) + lower
    return np.clip(angles, lower, upper)


def _parse_mimic_info(
    urdf_path: str,
) -> Dict[str, Tuple[str, float, float, float, float]]:
    """Extract mimic joint metadata from a URDF.

    Some SAPIEN versions do not automatically constrain mimic joints; in
    those cases mimic joints appear in ``get_active_joints()`` and must be
    driven manually using their reference joint's denormalized angle.

    Parameters
    ----------
    urdf_path : str
        Path to the ``.urdf`` file.

    Returns
    -------
    dict[str, tuple]
        Maps mimic joint name → ``(ref_name, multiplier, offset, lower, upper)``.
    """

    urdf = load_urdf_silent(urdf_path)
    info: Dict[str, Tuple[str, float, float, float, float]] = {}
    for joint in urdf.joints:
        if joint.mimic is None:
            continue
        lo = float(joint.limit.lower) if joint.limit is not None and joint.limit.lower is not None else float("-inf")
        hi = float(joint.limit.upper) if joint.limit is not None and joint.limit.upper is not None else float("inf")
        info[joint.name] = (
            str(joint.mimic.joint),
            float(joint.mimic.multiplier),
            float(joint.mimic.offset),
            lo,
            hi,
        )
    return info


def _build_sapien_trajectory(
    hand_name: str,
    joint_series: np.ndarray,
    robot: sapien.Articulation,
) -> np.ndarray:
    """Produce a (T, D_sapien) trajectory in SAPIEN's active-joint order.

    Two cases are handled:

    1. SAPIEN respects ``<mimic>`` tags and excludes mimic joints from
       ``get_active_joints()``.  Only independent DOFs need values.
    2. SAPIEN exposes mimic joints as active (older behaviour).  We detect
       this by checking whether the joint name is absent from
       ``model.dof_joints``; if it appears in the URDF's mimic table we
       compute its value from the reference joint's denormalized angle.

    Parameters
    ----------
    hand_name : str
        Key into ``HAND_CONFIGS``.
    joint_series : np.ndarray, shape=(T, D), dtype=float32
        Normalized trajectory in ``dof_joints`` order.
    robot : sapien.Articulation
        Loaded articulation whose active-joint list defines the target order.

    Returns
    -------
    np.ndarray, shape=(T, D_sapien), dtype=float32
        Physical joint trajectory in SAPIEN's active-joint order.

    Raises
    ------
    ValueError
        If an active joint name is unknown in both ``dof_joints`` and the
        URDF mimic table.
    """

    config = HAND_CONFIGS[hand_name]
    urdf_path = str(config["urdf_path"])
    model = HandKinematicsModel(
        hand_name=hand_name,
        urdf_path=urdf_path,
        root_link=str(config["root_link"]),
        tip_links=tuple(config["tip_links"]),
        wrist_link=str(config["wrist_link"]),
    )

    # Denormalize in model's dof_joints order → shape (T, D)
    angles = _denormalize_qpos(joint_series, model)

    # Name → column index in the denormalized array
    model_name_to_col: Dict[str, int] = {
        name: i for i, name in enumerate(model.dof_joints)
    }

    # Mimic fallback table (only used if SAPIEN exposes mimic joints)
    mimic_info = _parse_mimic_info(urdf_path)

    sapien_joints = robot.get_active_joints()
    n_sapien = len(sapien_joints)
    traj = np.zeros((joint_series.shape[0], n_sapien), dtype=np.float32)

    for sapien_idx, joint in enumerate(sapien_joints):
        jname = joint.get_name()
        col = model_name_to_col.get(jname)

        if col is not None:
            # Independent DOF — direct assignment
            traj[:, sapien_idx] = angles[:, col]
        elif jname in mimic_info:
            # SAPIEN exposed a mimic joint; compute from reference
            ref_name, mult, offset, lo, hi = mimic_info[jname]
            ref_col = model_name_to_col.get(ref_name)
            if ref_col is None:
                raise ValueError(
                    f"Mimic joint '{jname}' references '{ref_name}' which is "
                    f"also absent from dof_joints for '{hand_name}'."
                )
            traj[:, sapien_idx] = np.clip(
                angles[:, ref_col] * mult + offset, lo, hi
            ).astype(np.float32)
        else:
            raise ValueError(
                f"SAPIEN active joint '{jname}' is neither an independent DOF "
                f"nor a known mimic joint for '{hand_name}'.\n"
                f"  dof_joints  = {model.dof_joints}\n"
                f"  mimic_joints = {list(mimic_info.keys())}"
            )

    return traj


# ---------------------------------------------------------------------------
# Public API — single-hand loader
# ---------------------------------------------------------------------------


def load_hand_trajectory(
    hand_name: str,
    joint_series: np.ndarray,
    scene: sapien.Scene,
    base_pose: np.ndarray,
    per_frame_root_offsets: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> SapienHandInstance:
    """Load one hand URDF into the scene and prepare its replay trajectory.

    This is the SAPIEN counterpart of ``visualize.visualize_hand_motion``.
    Call this for every hand you want to visualize, then pass the resulting
    instances to :func:`run_sapien_replay`.

    Parameters
    ----------
    hand_name : str
        Key into ``HAND_CONFIGS``; selects the URDF and FK parameters.
    joint_series : np.ndarray, shape=(T, D), dtype=float32
        Normalized joint trajectory (same format as ``visualize_hand_motion``).
    scene : sapien.Scene
        Target scene returned by :func:`create_sapien_scene`.
    base_pose : np.ndarray, shape=(3,), dtype=float32
        Static world-frame XYZ position of this robot's base.
    per_frame_root_offsets : np.ndarray or None, shape=(T, 3), dtype=float32
        Per-frame additional translation (mirrors the Rerun argument).
    label : str or None
        Display label; defaults to ``hand_name``.

    Returns
    -------
    SapienHandInstance
        Loaded instance ready to be driven by :func:`run_sapien_replay`.
    """

    urdf_path = str(HAND_CONFIGS[hand_name]["urdf_path"])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_path)
    robot.set_root_pose(sapien.Pose(p=base_pose.tolist()))

    trajectory = _build_sapien_trajectory(hand_name, joint_series, robot)

    return SapienHandInstance(
        robot=robot,
        trajectory=trajectory,
        base_pose=base_pose.astype(np.float32),
        per_frame_root_offsets=(
            per_frame_root_offsets.astype(np.float32)
            if per_frame_root_offsets is not None
            else None
        ),
        label=label if label is not None else hand_name,
    )


# ---------------------------------------------------------------------------
# Public API — interactive replay loop
# ---------------------------------------------------------------------------


def run_sapien_replay(
    instances: List[SapienHandInstance],
    scene: sapien.Scene,
    frame_sleep: float = FRAME_SLEEP_SECONDS,
    loop: bool = True,
    camera_xyz: Optional[Tuple[float, float, float]] = None,
    camera_rpy: Optional[Tuple[float, float, float]] = None,
    window_title: str = "HandLatent SAPIEN Replay",
) -> None:
    """Open a SAPIEN viewer and replay joint trajectories interactively.

    Controls
    --------
    Space     : toggle play / pause
    →         : step one frame forward (while paused)
    ←         : step one frame backward (while paused)
    R         : restart from frame 0
    L         : toggle loop mode
    Q / Esc   : close viewer

    Parameters
    ----------
    instances : list[SapienHandInstance]
        Hand instances to replay.  All must have been loaded into ``scene``.
    scene : sapien.Scene
        Shared SAPIEN scene that holds the articulations.
    frame_sleep : float
        Seconds per frame during auto-play.
    loop : bool
        Whether playback restarts after the last frame.
    camera_xyz : tuple or None
        Camera world position ``(x, y, z)``.
    camera_rpy : tuple or None
        Camera Euler angles ``(roll, pitch, yaw)`` in radians.
    window_title : str
        Viewer window title.
    """

    if not instances:
        raise ValueError("No SapienHandInstance objects provided.")

    n_frames = max(inst.num_frames for inst in instances)

    viewer = scene.create_viewer()
    # Note: SAPIEN 3.0.x Viewer has no set_window_title; title is ignored.

    cam_xyz = camera_xyz if camera_xyz is not None else _DEFAULT_CAM_XYZ
    cam_rpy = camera_rpy if camera_rpy is not None else _DEFAULT_CAM_RPY
    viewer.set_camera_xyz(*cam_xyz)
    viewer.set_camera_rpy(*cam_rpy)

    playing: bool = True
    do_loop: bool = loop
    frame: int = 0
    last_time: float = time.perf_counter()

    # window.key_press(k) returns True exactly on the frame the key is first
    # pressed (rising-edge), so no manual debounce is needed.

    def _apply_all(f: int) -> None:
        for inst in instances:
            inst.apply_frame(f)
        scene.update_render()

    # Apply initial frame so the scene is not empty at startup
    _apply_all(0)

    while not viewer.closed:
        # --- rising-edge key handling (GLFW names, lowercase) ---
        w = viewer.window

        if w.key_press("space"):
            playing = not playing

        if w.key_press("l"):
            do_loop = not do_loop

        if w.key_press("r"):
            frame = 0
            _apply_all(frame)

        if not playing:
            if w.key_press("right"):
                frame = min(frame + 1, n_frames - 1)
                _apply_all(frame)
            if w.key_press("left"):
                frame = max(frame - 1, 0)
                _apply_all(frame)

        # --- auto-play (timer-driven, no sleep to keep viewer responsive) ---
        if playing:
            now = time.perf_counter()
            if now - last_time >= frame_sleep:
                last_time = now
                _apply_all(frame)
                frame += 1
                if frame >= n_frames:
                    if do_loop:
                        frame = 0
                    else:
                        playing = False
                        frame = n_frames - 1

        viewer.render()
