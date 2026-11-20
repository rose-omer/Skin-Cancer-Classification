import shutil

import numpy as np
import pytest
import torch

from skinlesion.model import (
    ONNX_FILE,
    create_model,
    export_onnx,
    features_and_logits,
    load_checkpoint,
    main,
    set_backbone_trainable,
)
from tests.conftest import TINY_MODEL, TINY_SIZE


def test_efficientnetv2_s_outputs_one_logit_per_image():
    model = create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False).eval()
    with torch.inference_mode():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_head_only_stage_freezes_backbone():
    model = create_model(TINY_MODEL, pretrained=False)
    set_backbone_trainable(model, False)
    head = {id(p) for p in model.get_classifier().parameters()}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == head
    set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_checkpoint_round_trip_keeps_outputs(tiny_checkpoint):
    model, cfg = load_checkpoint(tiny_checkpoint)
    reloaded, _ = load_checkpoint(tiny_checkpoint)
    x = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    with torch.inference_mode():
        assert torch.allclose(model(x), reloaded(x))
    assert cfg.threshold == pytest.approx(0.42)


def test_onnx_cli_writes_into_checkpoint_dir(tiny_checkpoint, tmp_path):
    pytest.importorskip("onnx")
    ckpt = tmp_path / "ckpt"
    shutil.copytree(tiny_checkpoint, ckpt, ignore=shutil.ignore_patterns(ONNX_FILE))
    main(["--export-onnx", str(ckpt)])
    assert (ckpt / ONNX_FILE).stat().st_size > 0


def test_onnx_export_matches_torch_for_any_batch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = create_model(TINY_MODEL, pretrained=False).eval()
    path = export_onnx(model, TINY_SIZE, tmp_path / "model.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        x = torch.randn(batch, 3, TINY_SIZE, TINY_SIZE)
        logit, feats = session.run(None, {"image": x.numpy()})
        with torch.inference_mode():
            ref_feats, ref_logit = features_and_logits(model, x)
        np.testing.assert_allclose(logit, ref_logit.numpy(), atol=1e-4)
        np.testing.assert_allclose(feats, ref_feats.numpy(), atol=1e-4)
