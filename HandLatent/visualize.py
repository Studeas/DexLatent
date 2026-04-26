"""Rerun visualization for normalized xarm hand trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
from urdf_parser_py import urdf as urdf_parser

from HandLatent.kinematics import ASSETS_DIR, HAND_CONFIGS, load_urdf_silent

TIMELINE_NAME: str = "step"
FRAME_SLEEP_SECONDS: float = 0.02

HAND_URDF_PATHS: Dict[str, str] = {
    name: str(config["urdf_path"]) for name, config in HAND_CONFIGS.items()
}


@dataclass
class RevoluteJoint:
    """Actuated revolute joint metadata for visualization playback.

    Parameters
    ----------
    name : str
        Joint name with shape=().
    axis : np.ndarray, shape=(3,), dtype=float32
        Joint axis in joint frame.
    lower : float
        Lower joint angle limit in radians with shape=().
    upper : float
        Upper joint angle limit in radians with shape=().
    link_path : str
        Rerun entity path for the child link with shape=().
    origin_translation : np.ndarray, shape=(3,), dtype=float32
        Joint origin translation in parent frame.
    origin_quaternion : np.ndarray, shape=(4,), dtype=float32
        Joint origin quaternion in ``[x, y, z, w]``.
    """

    name: str
    axis: np.ndarray
    lower: float
    upper: float
    link_path: str
    origin_translation: np.ndarray
    origin_quaternion: np.ndarray

    def angle_from_normalized(self, normalized: float) -> float:
        """Convert normalized joint command to angle in radians.

        Parameters
        ----------
        normalized : float
            Normalized command in ``[-1, 1]`` with shape=().

        Returns
        -------
        float
            Clamped joint angle in radians with shape=().
        """

        clipped = float(np.clip(normalized, -1.0, 1.0))
        return float(np.clip((clipped + 1.0) * 0.5 * (self.upper - self.lower) + self.lower, self.lower, self.upper))


@dataclass
class MimickedJoint:
    """Mimic joint metadata driven by one revolute reference joint.

    Parameters
    ----------
    name : str
        Mimic joint name with shape=().
    axis : np.ndarray, shape=(3,), dtype=float32
        Joint axis in joint frame.
    lower : float
        Lower angle limit with shape=().
    upper : float
        Upper angle limit with shape=().
    link_path : str
        Child link entity path with shape=().
    origin_translation : np.ndarray, shape=(3,), dtype=float32
        Origin translation.
    origin_quaternion : np.ndarray, shape=(4,), dtype=float32
        Origin quaternion in ``[x, y, z, w]``.
    reference : str
        Reference joint name with shape=().
    multiplier : float
        Mimic multiplier with shape=().
    offset : float
        Mimic additive offset with shape=().
    """

    name: str
    axis: np.ndarray
    lower: float
    upper: float
    link_path: str
    origin_translation: np.ndarray
    origin_quaternion: np.ndarray
    reference: str
    multiplier: float
    offset: float

    def angle_from_reference(self, reference_angle: float) -> float:
        """Compute mimic joint angle from reference angle.

        Parameters
        ----------
        reference_angle : float
            Reference joint angle in radians with shape=().

        Returns
        -------
        float
            Clamped mimic angle in radians with shape=().
        """

        return float(np.clip(reference_angle * self.multiplier + self.offset, self.lower, self.upper))


@dataclass
class PrismaticJoint:
    """Actuated prismatic joint metadata for visualization playback.

    Parameters
    ----------
    name : str
        Joint name with shape=().
    axis : np.ndarray, shape=(3,), dtype=float32
        Sliding axis in joint frame.
    lower : float
        Lower displacement limit in metres with shape=().
    upper : float
        Upper displacement limit in metres with shape=().
    link_path : str
        Rerun entity path for the child link with shape=().
    origin_translation : np.ndarray, shape=(3,), dtype=float32
        Joint origin translation in parent frame.
    origin_quaternion : np.ndarray, shape=(4,), dtype=float32
        Joint origin quaternion in ``[x, y, z, w]`` (constant for prismatic).
    """

    name: str
    axis: np.ndarray
    lower: float
    upper: float
    link_path: str
    origin_translation: np.ndarray
    origin_quaternion: np.ndarray

    def displacement_from_normalized(self, normalized: float) -> float:
        """Convert normalized command to displacement in metres.

        Parameters
        ----------
        normalized : float
            Normalized command in ``[-1, 1]`` with shape=().

        Returns
        -------
        float
            Clamped displacement in metres with shape=().
        """

        clipped = float(np.clip(normalized, -1.0, 1.0))
        return float(np.clip((clipped + 1.0) * 0.5 * (self.upper - self.lower) + self.lower, self.lower, self.upper))


@dataclass
class MimicPrismaticJoint:
    """Mimic prismatic joint driven by one independent prismatic reference.

    Parameters
    ----------
    name : str
        Joint name with shape=().
    axis : np.ndarray, shape=(3,), dtype=float32
        Sliding axis in joint frame.
    lower : float
        Lower displacement limit in metres with shape=().
    upper : float
        Upper displacement limit in metres with shape=().
    link_path : str
        Rerun entity path for the child link with shape=().
    origin_translation : np.ndarray, shape=(3,), dtype=float32
        Joint origin translation in parent frame.
    origin_quaternion : np.ndarray, shape=(4,), dtype=float32
        Joint origin quaternion in ``[x, y, z, w]``.
    reference : str
        Reference joint name with shape=().
    multiplier : float
        Mimic multiplier with shape=().
    offset : float
        Mimic additive offset in metres with shape=().
    """

    name: str
    axis: np.ndarray
    lower: float
    upper: float
    link_path: str
    origin_translation: np.ndarray
    origin_quaternion: np.ndarray
    reference: str
    multiplier: float
    offset: float

    def displacement_from_reference(self, reference_displacement: float) -> float:
        """Compute displacement from reference joint displacement.

        Parameters
        ----------
        reference_displacement : float
            Reference displacement in metres with shape=().

        Returns
        -------
        float
            Clamped displacement in metres with shape=().
        """

        return float(np.clip(reference_displacement * self.multiplier + self.offset, self.lower, self.upper))


def resolve_urdf_path(hand_name: str) -> str:
    """Resolve URDF path for one configured hand.

    Parameters
    ----------
    hand_name : str
        Hand name key with shape=().

    Returns
    -------
    str
        URDF path with shape=().
    """

    return HAND_URDF_PATHS[hand_name]


def _resolve_joint_axis(joint: urdf_parser.Joint) -> np.ndarray:
    """Resolve and normalize joint axis from URDF metadata.

    Parameters
    ----------
    joint : urdf_parser_py.urdf.Joint
        URDF joint object with shape=().

    Returns
    -------
    np.ndarray, shape=(3,), dtype=float32
        Normalized joint axis.
    """

    axis = np.array(joint.axis if joint.axis is not None else [0.0, 0.0, 1.0], dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(axis))
    return axis / norm


def _origin_to_numpy(joint: urdf_parser.Joint):
    """Extract origin translation and quaternion from a URDF joint.

    Parameters
    ----------
    joint : urdf_parser_py.urdf.Joint
        URDF joint object.

    Returns
    -------
    tuple
        - origin_translation : np.ndarray, shape=(3,), dtype=float32
        - origin_quaternion : np.ndarray, shape=(4,), dtype=float32 in ``[x,y,z,w]``
    """

    origin_translation = np.zeros(3, dtype=np.float32)
    origin_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if joint.origin is not None and joint.origin.xyz is not None:
        origin_translation = np.asarray(joint.origin.xyz, dtype=np.float32).reshape(3)
    if joint.origin is not None and joint.origin.rpy is not None:
        origin_quaternion = Rotation.from_euler("xyz", joint.origin.rpy).as_quat().astype(np.float32)
    return origin_translation, origin_quaternion


def discover_revolute_joints(urdf: urdf_parser.URDF, logger) -> List[RevoluteJoint]:
    """Extract actuated revolute joints from URDF.

    Parameters
    ----------
    urdf : urdf_parser_py.urdf.URDF
        Parsed URDF model with shape=().
    logger : object
        URDF logger exposing ``joint_entity_path``.

    Returns
    -------
    list[RevoluteJoint], shape=(M,)
        Ordered list of actuated joints.
    """

    joints: List[RevoluteJoint] = []
    for joint in urdf.joints:
        if joint.type != "revolute" or joint.limit is None or joint.axis is None or getattr(joint, "mimic", None) is not None:
            continue
        origin_translation, origin_quaternion = _origin_to_numpy(joint)
        joints.append(
            RevoluteJoint(
                name=joint.name,
                axis=_resolve_joint_axis(joint),
                lower=float(joint.limit.lower) if joint.limit.lower is not None else 0.0,
                upper=float(joint.limit.upper) if joint.limit.upper is not None else 0.0,
                link_path=logger.joint_entity_path(joint),
                origin_translation=origin_translation,
                origin_quaternion=origin_quaternion,
            )
        )
    return joints


def discover_mimic_joints(urdf: urdf_parser.URDF, logger) -> List[MimickedJoint]:
    """Extract mimic revolute joints from URDF.

    Parameters
    ----------
    urdf : urdf_parser_py.urdf.URDF
        Parsed URDF model with shape=().
    logger : object
        URDF logger exposing ``joint_entity_path``.

    Returns
    -------
    list[MimickedJoint], shape=(K,)
        Ordered list of mimic joints.
    """

    joints: List[MimickedJoint] = []
    for joint in urdf.joints:
        mimic = getattr(joint, "mimic", None)
        if joint.type != "revolute" or joint.limit is None or joint.axis is None or mimic is None:
            continue
        origin_translation, origin_quaternion = _origin_to_numpy(joint)
        joints.append(
            MimickedJoint(
                name=joint.name,
                axis=_resolve_joint_axis(joint),
                lower=float(joint.limit.lower) if joint.limit.lower is not None else 0.0,
                upper=float(joint.limit.upper) if joint.limit.upper is not None else 0.0,
                link_path=logger.joint_entity_path(joint),
                origin_translation=origin_translation,
                origin_quaternion=origin_quaternion,
                reference=str(mimic.joint),
                multiplier=float(mimic.multiplier),
                offset=float(mimic.offset),
            )
        )
    return joints


def discover_prismatic_joints(urdf: urdf_parser.URDF, logger) -> List[PrismaticJoint]:
    """Extract actuated prismatic joints from URDF.

    Parameters
    ----------
    urdf : urdf_parser_py.urdf.URDF
        Parsed URDF model with shape=().
    logger : object
        URDF logger exposing ``joint_entity_path``.

    Returns
    -------
    list[PrismaticJoint], shape=(P,)
        Ordered list of actuated prismatic joints.
    """

    joints: List[PrismaticJoint] = []
    for joint in urdf.joints:
        if joint.type != "prismatic" or joint.limit is None or joint.axis is None or getattr(joint, "mimic", None) is not None:
            continue
        origin_translation, origin_quaternion = _origin_to_numpy(joint)
        joints.append(
            PrismaticJoint(
                name=joint.name,
                axis=_resolve_joint_axis(joint),
                lower=float(joint.limit.lower) if joint.limit.lower is not None else 0.0,
                upper=float(joint.limit.upper) if joint.limit.upper is not None else 0.0,
                link_path=logger.joint_entity_path(joint),
                origin_translation=origin_translation,
                origin_quaternion=origin_quaternion,
            )
        )
    return joints


def discover_mimic_prismatic_joints(urdf: urdf_parser.URDF, logger) -> List[MimicPrismaticJoint]:
    """Extract mimic prismatic joints from URDF.

    Parameters
    ----------
    urdf : urdf_parser_py.urdf.URDF
        Parsed URDF model with shape=().
    logger : object
        URDF logger exposing ``joint_entity_path``.

    Returns
    -------
    list[MimicPrismaticJoint], shape=(Q,)
        Ordered list of mimic prismatic joints.
    """

    joints: List[MimicPrismaticJoint] = []
    for joint in urdf.joints:
        mimic = getattr(joint, "mimic", None)
        if joint.type != "prismatic" or joint.limit is None or joint.axis is None or mimic is None:
            continue
        origin_translation, origin_quaternion = _origin_to_numpy(joint)
        joints.append(
            MimicPrismaticJoint(
                name=joint.name,
                axis=_resolve_joint_axis(joint),
                lower=float(joint.limit.lower) if joint.limit.lower is not None else 0.0,
                upper=float(joint.limit.upper) if joint.limit.upper is not None else 0.0,
                link_path=logger.joint_entity_path(joint),
                origin_translation=origin_translation,
                origin_quaternion=origin_quaternion,
                reference=str(mimic.joint),
                multiplier=float(mimic.multiplier),
                offset=float(mimic.offset),
            )
        )
    return joints


def scale_joint_values(joint_series: np.ndarray, joints: Sequence[RevoluteJoint]) -> np.ndarray:
    """Convert normalized trajectories to revolute joint angles.

    Parameters
    ----------
    joint_series : np.ndarray, shape=(T, M), dtype=float32
        Normalized trajectory.
    joints : Sequence[RevoluteJoint], shape=(M,)
        Revolute joints matching the trajectory dimension.

    Returns
    -------
    np.ndarray, shape=(T, M), dtype=float32
        Angle trajectory in radians.
    """

    return np.asarray(
        [[joint.angle_from_normalized(float(value)) for joint, value in zip(joints, frame)] for frame in joint_series],
        dtype=np.float32,
    )


def visualize_hand_motion(
    hand_name: str,
    joint_series: np.ndarray,
    recording_name: str,
    recording: Optional[rr.RecordingStream] = None,
    entity_path_prefix: Optional[str] = None,
    per_frame_root_offsets: Optional[np.ndarray] = None,
) -> None:
    """Visualize one normalized trajectory in Rerun using URDF geometry.

    Parameters
    ----------
    hand_name : str
        Hand name key with shape=().
    joint_series : np.ndarray, shape=(T, D), dtype=float32
        Normalized trajectory where ``D`` is the total actuated DOF count
        (revolute joints first, then prismatic joints, matching ``dof_joints`` order).
    recording_name : str
        Rerun recording name with shape=().
    recording : rr.RecordingStream or None
        Existing recording stream.
    entity_path_prefix : str or None
        Optional path prefix for entities.
    per_frame_root_offsets : np.ndarray or None, shape=(T, 3), dtype=float32
        Optional root translation offsets per frame.

    Returns
    -------
    None
        Logs dynamic transforms to Rerun.
    """

    if recording is None:
        rr.init(recording_name, spawn=True)
        recording = rr.get_global_data_recording()

    recording.set_time(TIMELINE_NAME, duration=0.0)
    urdf_path = resolve_urdf_path(hand_name)
    urdf = load_urdf_silent(urdf_path)
    prefix = entity_path_prefix if entity_path_prefix is not None else urdf.get_root()

    from rerun_loader_urdf import URDFLogger

    logger = URDFLogger(urdf_path, prefix)
    logger.log(recording)

    revolute_joints = discover_revolute_joints(urdf, logger)
    mimic_joints = discover_mimic_joints(urdf, logger)
    prismatic_joints = discover_prismatic_joints(urdf, logger)
    mimic_prismatic_joints = discover_mimic_prismatic_joints(urdf, logger)

    n_rev = len(revolute_joints)
    n_pris = len(prismatic_joints)

    # Columns 0:n_rev → revolute arm joints; n_rev:n_rev+n_pris → prismatic finger joints.
    # This matches the dof_joints BFS order produced by HandKinematicsModel._parse_urdf().
    revolute_series = joint_series[:, :n_rev]
    prismatic_series = joint_series[:, n_rev : n_rev + n_pris] if n_pris > 0 else np.zeros((joint_series.shape[0], 0), dtype=np.float32)

    scaled_revolute = scale_joint_values(revolute_series, revolute_joints)

    for step, rev_angles in enumerate(scaled_revolute):
        recording.set_time(TIMELINE_NAME, duration=step * FRAME_SLEEP_SECONDS)
        if per_frame_root_offsets is not None:
            offset = np.asarray(per_frame_root_offsets[step], dtype=np.float32).reshape(3)
            recording.log(prefix, rr.Transform3D.from_fields(translation=offset.tolist()))

        evaluated_angles: Dict[str, float] = {}

        # --- revolute joints ---
        for joint, angle in zip(revolute_joints, rev_angles):
            evaluated_angles[joint.name] = float(angle)
            quaternion = (Rotation.from_quat(joint.origin_quaternion) * Rotation.from_rotvec(joint.axis * float(angle))).as_quat()
            recording.log(
                joint.link_path,
                rr.Transform3D.from_fields(
                    translation=joint.origin_translation,
                    quaternion=quaternion,
                ),
            )

        # --- mimic revolute joints ---
        for mimic_joint in mimic_joints:
            reference_angle = evaluated_angles[mimic_joint.reference]
            mimic_angle = mimic_joint.angle_from_reference(reference_angle)
            evaluated_angles[mimic_joint.name] = mimic_angle
            quaternion = (Rotation.from_quat(mimic_joint.origin_quaternion) * Rotation.from_rotvec(mimic_joint.axis * mimic_angle)).as_quat()
            recording.log(
                mimic_joint.link_path,
                rr.Transform3D.from_fields(
                    translation=mimic_joint.origin_translation,
                    quaternion=quaternion,
                ),
            )

        # --- prismatic joints (translation, not rotation) ---
        evaluated_displacements: Dict[str, float] = {}
        for pjoint, norm_val in zip(prismatic_joints, prismatic_series[step]):
            displacement = pjoint.displacement_from_normalized(float(norm_val))
            evaluated_displacements[pjoint.name] = displacement
            translation = pjoint.origin_translation + pjoint.axis * displacement
            recording.log(
                pjoint.link_path,
                rr.Transform3D.from_fields(
                    translation=translation.tolist(),
                    quaternion=pjoint.origin_quaternion,
                ),
            )

        # --- mimic prismatic joints ---
        for pmimic in mimic_prismatic_joints:
            ref_disp = evaluated_displacements.get(pmimic.reference, 0.0)
            displacement = pmimic.displacement_from_reference(ref_disp)
            evaluated_displacements[pmimic.name] = displacement
            translation = pmimic.origin_translation + pmimic.axis * displacement
            recording.log(
                pmimic.link_path,
                rr.Transform3D.from_fields(
                    translation=translation.tolist(),
                    quaternion=pmimic.origin_quaternion,
                ),
            )
