import torch

from HandLatent.kinematics import MultiHandDifferentiableFK
from HandLatent.model import compute_semantic_grasp_loss


def test_semantic_loss_detects_opposer_changes_with_same_centroid() -> None:
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, -0.5, 0.0]]],
        dtype=torch.float32,
    )

    distance, direction, weight = compute_semantic_grasp_loss(source, target, 12.0)

    assert distance.shape == (1, 2)
    assert direction.shape == (1, 2)
    assert weight.shape == (1, 2)
    assert distance.mean() > 0.0


def test_semantic_loss_uses_shared_pairs_for_five_to_three_with_gradient() -> None:
    source = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    distance, _, _ = compute_semantic_grasp_loss(source, target, 12.0)
    loss = distance.mean()
    loss.backward()

    assert distance.shape == (1, 2)
    assert target.grad is not None
    assert torch.linalg.norm(target.grad) > 0.0


def test_semantic_loss_matches_gripper_to_average_hand_aperture() -> None:
    source = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 7.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )

    distance, direction, weight = compute_semantic_grasp_loss(source, target, 12.0)

    assert distance.shape == (1,)
    assert torch.allclose(distance, torch.tensor([4.0]))
    assert direction.item() == 0.0
    assert weight.item() >= 0.25


def test_semantic_weight_floor_keeps_large_apertures_weighted() -> None:
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    _, _, weight = compute_semantic_grasp_loss(source, target, 12.0, semantic_weight_floor=0.25)

    assert torch.allclose(weight, torch.tensor([0.25]))


def test_panda_gripper_fk_uses_tip_links_with_contact_offset() -> None:
    registry = MultiHandDifferentiableFK(["xarm7_panda_gripper_right"])
    model = registry.models["xarm7_panda_gripper_right"]

    assert model.tip_links == ["panda_leftfinger_tip", "panda_rightfinger_tip"]
    assert model.tip_count() == 2
    assert model.dof_count() - 7 == 1

    closed = torch.zeros(model.dof_count(), dtype=torch.float32)
    opened = closed.clone()
    opened[-1] = 1.0
    closed_tips, wrist_pose = model.forward_with_wrist_pose(closed)
    opened_tips = model.forward(opened)

    closed_gap = torch.linalg.norm(closed_tips[0] - closed_tips[1])
    opened_gap = torch.linalg.norm(opened_tips[0] - opened_tips[1])
    closed_tips_h = torch.cat([closed_tips, torch.ones(2, 1, dtype=closed_tips.dtype)], dim=1)
    wrist_frame_tips = (torch.linalg.inv(wrist_pose) @ closed_tips_h.T).T[:, :3]

    assert opened_gap > closed_gap
    assert torch.isclose(wrist_frame_tips[:, 2].mean(), torch.tensor(0.008 + 0.0584 + 0.04525), atol=1.0e-5)
