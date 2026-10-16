"""
test.py
Automated unit tests verifying preprocessing, model inference,
typographic vulnerability, and defense mitigation effectiveness.
Run with: pytest test.py
"""

import pytest
from PIL import Image

from preprocess import (
    generate_synthetic_apple,
    overlay_text_on_image,
    mask_typographic_region,
)
from research import (
    load_clip_pipeline,
    get_clip_confidence,
    get_clip_confidence_orthogonal_defense,
    get_clip_confidence_cap_pruning,
)


@pytest.fixture(scope="session")
def clip_components():
    """Initializes and shares CLIP model and processor across tests."""
    processor, model = load_clip_pipeline()
    return processor, model


@pytest.fixture
def synthetic_image():
    """Generates a clean synthetic apple image fixture."""
    return generate_synthetic_apple()


def test_synthetic_image_generation(synthetic_image):
    assert isinstance(synthetic_image, Image.Image)
    assert synthetic_image.size == (384, 384)
    assert synthetic_image.mode == "RGB"


def test_typographic_overlay(synthetic_image):
    attacked_img, bbox = overlay_text_on_image(
        synthetic_image, text="Test", font_size=32
    )
    assert isinstance(attacked_img, Image.Image)
    assert len(bbox) == 4
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]


def test_visual_masking(synthetic_image):
    attacked_img, bbox = overlay_text_on_image(
        synthetic_image, text="MaskMe", font_size=40
    )
    defended_img = mask_typographic_region(attacked_img, bbox)
    assert isinstance(defended_img, Image.Image)
    assert defended_img.size == synthetic_image.size


def test_baseline_clean_prediction(clip_components, synthetic_image):
    processor, model = clip_components
    labels = ["a photo of a apple", "a photo of a orange"]
    scores = get_clip_confidence(processor, model, synthetic_image, labels)

    assert "a photo of a apple" in scores
    assert "a photo of a orange" in scores
    assert scores["a photo of a apple"] > scores["a photo of a orange"]


def test_typographic_vulnerability(clip_components, synthetic_image):
    processor, model = clip_components
    labels = ["a photo of a apple", "a photo of a orange"]

    attacked_img, _ = overlay_text_on_image(
        synthetic_image, text="Orange", font_size=64
    )
    scores = get_clip_confidence(processor, model, attacked_img, labels)

    assert scores["a photo of a orange"] > 0.30


def test_visual_defense_recovery(clip_components, synthetic_image):
    processor, model = clip_components
    labels = ["a photo of a apple", "a photo of a orange"]

    attacked_img, bbox = overlay_text_on_image(
        synthetic_image, text="Orange", font_size=64
    )
    defended_img = mask_typographic_region(attacked_img, bbox)

    defended_scores = get_clip_confidence(processor, model, defended_img, labels)
    assert defended_scores["a photo of a apple"] > defended_scores["a photo of a orange"]


def test_cap_pruning_defense(clip_components, synthetic_image):
    processor, model = clip_components
    labels = ["a photo of a apple", "a photo of a orange"]

    attacked_img, _ = overlay_text_on_image(
        synthetic_image, text="Orange", font_size=48
    )
    defended_scores = get_clip_confidence_cap_pruning(
        processor, model, attacked_img, labels
    )
    assert sum(defended_scores.values()) == pytest.approx(1.0, abs=1e-3)


def test_orthogonal_defense(clip_components, synthetic_image):
    processor, model = clip_components
    labels = ["a photo of a apple", "a photo of a orange"]

    attacked_img, _ = overlay_text_on_image(
        synthetic_image, text="Orange", font_size=48
    )
    defended_scores = get_clip_confidence_orthogonal_defense(
        processor, model, attacked_img, labels, distractor_idx=1
    )
    assert sum(defended_scores.values()) == pytest.approx(1.0, abs=1e-3)