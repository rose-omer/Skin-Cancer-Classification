import numpy as np
import pytest
import torch

from skinlesion.explain import grad_cam, overlay, save_gallery
from skinlesion.model import create_model
from tests.conftest import TINY_MODEL, TINY_SIZE, fake_image


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return create_model(TINY_MODEL, pretrained=False).eval()


def test_grad_cam_matches_input_size_and_unit_range(model):
    cam = grad_cam(model, torch.randn(1, 3, TINY_SIZE, TINY_SIZE))
    assert cam.shape == (TINY_SIZE, TINY_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_grad_cam_rejects_batches(model):
    with pytest.raises(ValueError):
        grad_cam(model, torch.randn(2, 3, TINY_SIZE, TINY_SIZE))


def test_grad_cam_leaves_no_gradients_on_input(model):
    x = torch.randn(1, 3, TINY_SIZE, TINY_SIZE)
    grad_cam(model, x)
    assert x.grad is None


def test_grad_cam_never_writes_parameter_gradients(model):
    """Paylaşılan model eşzamanlı isteklerde kullanılır; .grad birikmemeli."""
    for p in model.parameters():
        p.grad = None
    grad_cam(model, torch.randn(1, 3, TINY_SIZE, TINY_SIZE))
    assert all(p.grad is None for p in model.parameters())


def test_grad_cam_is_thread_safe(model):
    from concurrent.futures import ThreadPoolExecutor

    xs = [
        torch.randn(1, 3, TINY_SIZE, TINY_SIZE, generator=torch.Generator().manual_seed(i))
        for i in range(4)
    ]
    expected = [grad_cam(model, x) for x in xs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda x: grad_cam(model, x), xs))
    for a, b in zip(expected, got, strict=True):
        assert np.allclose(a, b, atol=1e-5)


def test_overlay_keeps_image_size_and_alpha_zero_is_identity():
    img = fake_image(0, 0, size=40)
    cam = np.random.default_rng(0).random((TINY_SIZE, TINY_SIZE))
    assert overlay(img, cam).size == img.size
    assert np.array_equal(np.asarray(overlay(img, cam, alpha=0.0)), np.asarray(img))
    with pytest.raises(ValueError):
        overlay(img, cam, alpha=1.5)


def test_save_gallery_writes_png(tmp_path):
    images = [fake_image(i, i % 2) for i in range(5)]
    out = save_gallery(images, [f"#{i}" for i in range(5)], tmp_path / "g.png", cols=4)
    assert out.stat().st_size > 0
