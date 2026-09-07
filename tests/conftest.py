import pytest
import torch

from bcrnet import ModelConfig


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def cfg():
    return ModelConfig(
        stem_channels=8,
        backbone_channels=(16, 24, 32, 48),
        backbone_depths=(1, 1, 1, 1),
        width=16,
        heads=4,
        budget=2,
    )


@pytest.fixture
def targets():
    return [
        {
            "boxes": torch.tensor([[10.0, 10.0, 16.0, 17.0], [40.0, 20.0, 75.0, 50.0]]),
            "labels": torch.tensor([0, 0]),
        },
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
    ]
