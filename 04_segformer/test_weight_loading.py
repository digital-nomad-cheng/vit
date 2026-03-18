"""Test that HuggingFace SegFormer-B0 pretrained weights load correctly.

Downloads the checkpoint once (cached by huggingface_hub) and verifies
that every key maps without missing or unexpected entries.
"""

import torch
from segformer import segformer_b0, _map_hf_key, load_pretrained_segformer_b0


def test_key_mapping_covers_all_hf_keys():
    """Every HF checkpoint key should map to a valid model key."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id="nvidia/segformer-b0-finetuned-ade-512-512",
        filename="pytorch_model.bin",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = segformer_b0(num_classes=150)
    model_keys = set(model.state_dict().keys())

    mapped_keys = {}
    unmapped_hf_keys = []
    for hf_key in state_dict:
        our_key = _map_hf_key(hf_key)
        if our_key is None:
            unmapped_hf_keys.append(hf_key)
        else:
            mapped_keys[our_key] = hf_key

    # No HF key should be completely unmapped (return None)
    assert not unmapped_hf_keys, (
        f"HF keys returned None from _map_hf_key:\n"
        + "\n".join(f"  {k}" for k in unmapped_hf_keys)
    )

    # Every mapped key should exist in the model
    unexpected = set(mapped_keys.keys()) - model_keys
    assert not unexpected, (
        f"Mapped keys not found in model:\n"
        + "\n".join(f"  {k} (from {mapped_keys[k]})" for k in sorted(unexpected))
    )

    # Every model key should be covered by the checkpoint
    missing = model_keys - set(mapped_keys.keys())
    assert not missing, (
        f"Model keys not covered by checkpoint:\n"
        + "\n".join(f"  {k}" for k in sorted(missing))
    )


def test_load_pretrained_no_errors():
    """load_pretrained_segformer_b0 should load with zero missing/unexpected keys."""
    model = segformer_b0(num_classes=150)
    model = load_pretrained_segformer_b0(model, num_classes=150)

    # Verify the model runs a forward pass after loading
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 150, 128, 128), f"Unexpected output shape: {out.shape}"


def test_weight_shapes_match():
    """All loaded weights should have matching shapes (no reshape errors)."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id="nvidia/segformer-b0-finetuned-ade-512-512",
        filename="pytorch_model.bin",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = segformer_b0(num_classes=150)
    model_state = model.state_dict()

    # Build mapped dict (mirrors load_pretrained_segformer_b0 logic)
    mapped = {}
    for hf_key, value in state_dict.items():
        key = _map_hf_key(hf_key)
        if key is not None:
            mapped[key] = value

    # Reshape decoder linear projections (2D → 4D)
    for key in mapped:
        if "decoder.linear_projs" in key and "weight" in key and mapped[key].ndim == 2:
            mapped[key] = mapped[key].unsqueeze(-1).unsqueeze(-1)

    # Check shapes
    mismatches = []
    for key in mapped:
        if key in model_state:
            if mapped[key].shape != model_state[key].shape:
                mismatches.append(
                    f"  {key}: checkpoint {mapped[key].shape} vs model {model_state[key].shape}"
                )

    assert not mismatches, (
        f"Shape mismatches:\n" + "\n".join(mismatches)
    )


if __name__ == "__main__":
    print("Test 1: Key mapping covers all HF keys...")
    test_key_mapping_covers_all_hf_keys()
    print("  PASSED")

    print("Test 2: Weight shapes match...")
    test_weight_shapes_match()
    print("  PASSED")

    print("Test 3: Load pretrained with no errors...")
    test_load_pretrained_no_errors()
    print("  PASSED")

    print("\nAll tests passed!")
