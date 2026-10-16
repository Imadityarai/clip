"""
research.py
CLIP zero-shot inference, typographic scaling experiments, and defense algorithms.
Includes unwrap_tensor to maintain compatibility across Transformers v4 and v5.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from preprocess import (
    load_image,
    generate_synthetic_apple,
    overlay_text_on_image,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip_pipeline() -> Tuple[CLIPProcessor, CLIPModel]:
    """Loads and caches the pre-trained CLIP model and processor."""
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model


def unwrap_tensor(output) -> torch.Tensor:
    """
    Extracts a raw PyTorch Tensor from transformers wrapper dataclasses
    (handles BaseModelOutputWithPooling, CLIPOutput, and raw Tensors).
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    return output


# -------------------------------------------------------------------------
# INFERENCE & DEFENSE METHODS
# -------------------------------------------------------------------------

def get_clip_confidence(
    processor: CLIPProcessor,
    model: CLIPModel,
    image_input,
    text_labels: List[str],
    defense_prefix: str = "",
) -> Dict[str, float]:
    """Standard baseline CLIP zero-shot classification with optional prompt prefixing."""
    image = load_image(image_input) or generate_synthetic_apple()

    prompts = [f"{defense_prefix} {lbl}".strip() for lbl in text_labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(text_labels)}


def get_clip_confidence_orthogonal_defense(
    processor: CLIPProcessor,
    model: CLIPModel,
    image_input,
    text_labels: List[str],
    distractor_idx: int = 1,
    alpha: float = 0.85,
) -> Dict[str, float]:
    """
    Defense 3: Orthogonal Subspace Projection (CMOP / LCAR).
    Projects out the typographic distractor text vector from the visual embedding.
    """
    image = load_image(image_input) or generate_synthetic_apple()

    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        image_embeds = unwrap_tensor(model.get_image_features(inputs["pixel_values"]))
        text_embeds = unwrap_tensor(model.get_text_features(inputs["input_ids"]))

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        distractor_vec = text_embeds[distractor_idx : distractor_idx + 1]

        # Orthogonal component subtraction
        dot = torch.sum(image_embeds * distractor_vec, dim=-1, keepdim=True)
        defended_image_embeds = image_embeds - (alpha * dot * distractor_vec)
        defended_image_embeds = defended_image_embeds / defended_image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (defended_image_embeds @ text_embeds.T)
        probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(text_labels)}


def get_clip_confidence_cap_pruning(
    processor: CLIPProcessor,
    model: CLIPModel,
    image_input,
    text_labels: List[str],
    temperature: float = 0.07,
    prune_percentile: float = 75.0,
) -> Dict[str, float]:
    """
    Defense 4: ViT Spatial Token Suppression (CAP-Pruning).
    Suppresses ViT visual tokens exhibiting high cross-modal alignment with text.
    """
    image = load_image(image_input) or generate_synthetic_apple()

    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = model.vision_model(inputs["pixel_values"], output_hidden_states=True)
        last_hidden_state = vision_outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0:1, :]
        patch_tokens = last_hidden_state[:, 1:, :]

        patch_tokens_proj = (
            model.visual_projection(patch_tokens)
            if hasattr(model, "visual_projection")
            else patch_tokens
        )
        cls_token_proj = (
            model.visual_projection(cls_token)
            if hasattr(model, "visual_projection")
            else cls_token
        )

        patch_norm = patch_tokens_proj / patch_tokens_proj.norm(dim=-1, keepdim=True)

        text_embeds = unwrap_tensor(model.get_text_features(inputs["input_ids"]))
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Cross-modal patch similarity
        patch_sim = torch.matmul(patch_norm, text_embeds.T) / temperature
        max_sim_per_patch, _ = torch.max(patch_sim, dim=-1)

        # Prune text-heavy visual patches using dynamic quantile threshold
        cutoff = torch.quantile(max_sim_per_patch, prune_percentile / 100.0)
        patch_weights = torch.sigmoid(-(max_sim_per_patch - cutoff)).unsqueeze(-1)

        weighted_patches = patch_norm * patch_weights
        defended_embed = cls_token_proj + weighted_patches.sum(dim=1, keepdim=True)
        defended_embed = defended_embed.squeeze(1)
        defended_embed = defended_embed / defended_embed.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (defended_embed @ text_embeds.T)
        probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

    return {label: float(probs[i]) for i, label in enumerate(text_labels)}


# -------------------------------------------------------------------------
# BENCHMARK SUITE
# -------------------------------------------------------------------------

def run_font_scaling_study(
    processor: CLIPProcessor,
    model: CLIPModel,
    base_image,
    target_class: str = "a photo of a apple",
    adversarial_text: str = "Orange",
    adversarial_class: str = "a photo of a orange",
    font_scales: List[int] = None,
    save_plot_path: str = "scaling_study.png",
):
    """Measures and plots CLIP confidence changes across multiple text sizes."""
    if font_scales is None:
        font_scales = [14, 24, 36, 48, 64, 84]

    labels = [target_class, adversarial_class]
    target_scores, adv_scores = [], []

    print("[INFO] Running font scaling benchmark...")
    for scale in font_scales:
        attacked_img, _ = overlay_text_on_image(
            base_image, text=adversarial_text, font_size=scale
        )
        scores = get_clip_confidence(processor, model, attacked_img, labels)
        target_scores.append(scores[target_class])
        adv_scores.append(scores[adversarial_class])
        print(f"Scale {scale}pt -> True: {scores[target_class]:.4f} | Adv: {scores[adversarial_class]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(font_scales, target_scores, "o-", color="#2ecc71", label=f"Target: '{target_class}'")
    ax.plot(font_scales, adv_scores, "s-", color="#e74c3c", label=f"Adversarial: '{adversarial_class}'")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="Decision Threshold (0.5)")
    ax.set_title("Effect of Typographic Font Scale on CLIP Predictions")
    ax.set_xlabel("Rendered Font Scale (pt)")
    ax.set_ylabel("Softmax Confidence")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Scaling plot saved to: {save_plot_path}")


if __name__ == "__main__":
    proc, m = load_clip_pipeline()
    clean_img = generate_synthetic_apple()
    run_font_scaling_study(proc, m, clean_img)