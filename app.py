"""
app.py
Interactive Streamlit Dashboard demonstrating CLIP typographic attacks and defenses.
Fully dynamic text overlay: user text input updates all tabs, renders, and charts.
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from preprocess import (
    load_image,
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

# -------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & STYLING
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="CLIP Typographic Vulnerability & Defense Lab",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #4e73df;
        margin-bottom: 12px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Cache model pipeline
@st.cache_resource(show_spinner="Loading CLIP ViT-B/32 backbone...")
def get_cached_models():
    return load_clip_pipeline()

processor, model = get_cached_models()

# -------------------------------------------------------------------------
# SIDEBAR CONTROLS (UNIFIED DYNAMIC INPUTS)
# -------------------------------------------------------------------------
st.sidebar.title("🎛️ Experiment Parameters")

# 1. Image Source Selection
source_option = st.sidebar.radio(
    "Sample Image Source:",
    ["Synthetic Sample (Apple)", "Upload Custom Image", "Local File Path"],
)

base_image = None
if source_option == "Synthetic Sample (Apple)":
    base_image = generate_synthetic_apple()

elif source_option == "Upload Custom Image":
    uploaded = st.sidebar.file_uploader(
        "Upload Image (PNG, JPG, WEBP)", 
        type=["png", "jpg", "jpeg", "webp"]
    )
    if uploaded is not None:
        base_image = load_image(uploaded)
        if base_image is None:
            st.sidebar.error("Could not parse uploaded file. Using synthetic fallback.")
            base_image = generate_synthetic_apple()
        else:
            st.sidebar.success(f"Loaded image: {base_image.width}x{base_image.height}")
    else:
        st.sidebar.info("Upload an image above, or using the default sample.")
        base_image = generate_synthetic_apple()

else:  # Local File Path
    path = st.sidebar.text_input("Local Image Path:", "/content/Screenshot 2026-09-03 004546.png")
    base_image = load_image(path)
    if base_image is None:
        st.sidebar.warning("Path not found. Using synthetic apple fallback.")
        base_image = generate_synthetic_apple()

if base_image is None:
    base_image = generate_synthetic_apple()

st.sidebar.divider()
st.sidebar.subheader("🏷️ Object & Typographic Labels")

# 2. Text Prompts & Typography (Dynamic Binding)
true_concept = st.sidebar.text_input("True Visual Object:", value="apple")
adv_text_global = st.sidebar.text_input("Adversarial Text to Imprint on Image:", value="Orange")

# Format candidate prompts
true_label = f"a photo of a {true_concept.strip()}"
misleading_label = f"a photo of a {adv_text_global.strip().lower()}"
global_labels = [true_label, misleading_label]

st.sidebar.caption(f"**Evaluating candidate classes:**\n- `{true_label}`\n- `{misleading_label}`")

# -------------------------------------------------------------------------
# NAVIGATION TABS
# -------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 1. Research Study & Findings",
    "🛡️ 2. Defense Mechanisms Architecture",
    "🧪 3. Live Interactive Comparison",
])

# =========================================================================
# TAB 1: EMPIRICAL RESEARCH FINDINGS
# =========================================================================
with tab1:
    st.header("Vulnerability Analysis: Clean vs. Typographic Attack")
    st.markdown(
        f"""
        Multimodal models like CLIP project vision and language embeddings into a **shared metric space**.
        Because web-scale training data contains annotated imagery, CLIP has developed **typographic bias**: 
        when conflicting text (like **"{adv_text_global}"**) is written onto an object, optical character features 
        frequently overwhelm the physical visual geometry.
        """
    )

    c1, c2 = st.columns(2)
    clean_scores = get_clip_confidence(processor, model, base_image, global_labels)
    
    # Dynamically overlay user's adversarial text
    attacked_img, demo_bbox = overlay_text_on_image(base_image, text=adv_text_global, font_size=52)
    attacked_scores = get_clip_confidence(processor, model, attacked_img, global_labels)

    with c1:
        st.subheader("Clean Baseline (No Typography)")
        st.image(base_image, width=320, caption="True Visual Object")
        st.write(f"🍏 **{true_label}:** `{clean_scores[true_label]:.4f}` ({clean_scores[true_label]*100:.1f}%)")
        st.write(f"🍊 **{misleading_label}:** `{clean_scores[misleading_label]:.4f}` ({clean_scores[misleading_label]*100:.1f}%)")

    with c2:
        st.subheader(f"Attacked Image (Overlaid '{adv_text_global}')")
        st.image(attacked_img, width=320, caption=f"Visual Object + '{adv_text_global}' Text")
        st.write(f"🍏 **{true_label}:** `{attacked_scores[true_label]:.4f}` ({attacked_scores[true_label]*100:.1f}%)")
        st.write(f"🍊 **{misleading_label}:** `{attacked_scores[misleading_label]:.4f}` ({attacked_scores[misleading_label]*100:.1f}%)")

    st.divider()
    st.subheader(f"Impact of Font Scale on '{adv_text_global}' Attack")
    st.markdown("Larger text spans more ViT patches, causing attention heads to prioritize text over visual features.")

    if st.button("▶️ Run Font Scale Sweep Experiment"):
        scales = [14, 28, 42, 56, 70, 84]
        t_scores, a_scores = [], []
        prog = st.progress(0)
        for idx, s in enumerate(scales):
            img_s, _ = overlay_text_on_image(base_image, text=adv_text_global, font_size=s)
            res = get_clip_confidence(processor, model, img_s, global_labels)
            t_scores.append(res[true_label])
            a_scores.append(res[misleading_label])
            prog.progress((idx + 1) / len(scales))

        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.plot(scales, t_scores, "o-", color="#2ecc71", linewidth=2.5, label=f"True: '{true_label}'")
        ax.plot(scales, a_scores, "s-", color="#e74c3c", linewidth=2.5, label=f"Adversarial: '{misleading_label}'")
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Decision Threshold (50%)")
        ax.set_title(f"CLIP Confidence vs. '{adv_text_global}' Font Scale", fontsize=12, fontweight="bold")
        ax.set_xlabel("Rendered Font Scale (pt)")
        ax.set_ylabel("Softmax Confidence Score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        st.pyplot(fig)


# =========================================================================
# TAB 2: TECHNICAL DEFENSE EXPLANATIONS
# =========================================================================
with tab2:
    st.header("Technical Explanations of Primary Defense Mechanisms")
    st.markdown(
        """
        To prevent typographic attacks from hijacking vision-language embeddings, defenses 
        intervene either at the **Visual Input Preprocessing Stage** or the **Language Conditioning Stage**.
        """
    )

    st.divider()

    # ----------------- DEFENSE 1 -----------------
    st.subheader("🛡️ Defense Mechanism 1: Text Masking & Bounding Box Erasure")
    st.markdown("**Domain:** Visual Pre-processing (Pixel Space)")

    col_d1_text, col_d1_diag = st.columns([1, 1.2])

    with col_d1_text:
        st.markdown(
            f"""
            #### How It Works:
            1. **Detection:** An OCR detector (e.g., CRAFT, DBNet, or EasyOCR) scans the input image for typographical character edges.
            2. **Bounding Box Delineation:** Coordinates $[x_0, y_0, x_1, y_1]$ encompassing the text **"{adv_text_global}"** are isolated.
            3. **Pixel Erasure & Neutral Inpainting:** The bounding box region is filled with a neutral neutral-gray/background color before being passed to CLIP's visual tokenizer.
            
            #### Mathematical Formulation:
            Let $I(u, v)$ be the input pixel array and $B$ be the detected bounding box:
            $$I_{{\\text{{defended}}}}(u, v) = \\begin{{cases}} \\mathbf{{c}}_{{\\text{{neutral}}}} & \\text{{if }} (u, v) \\in B \\\\ I(u, v) & \\text{{otherwise}} \\end{{cases}}$$
            
            #### Key Benefit:
            Because the text is sanitized at the raw pixel stage, the Vision Transformer (ViT) receives **zero typographic tokens**.
            """
        )

    with col_d1_diag:
        st.markdown("**Behind the Scenes Architecture:**")
        st.code(
            f"""
   [ Attacked Image ] (Contains "{adv_text_global}" on Object)
           │
           ▼
   [ Text Detection / OCR ] ──► Locates BBox: [x0, y0, x1, y1]
           │
           ▼
   [ Spatial Inpainting ]   ──► Replaces box pixels with neutral fill
           │
           ▼
   [ Sanitized Image ]      ──► 0% text visual tokens reach ViT
           │
           ▼
   [ ViT Vision Transformer ]
           │
           ▼
   [ Purified Image Embedding v ] ──► Matches "{true_label}" ✅
            """,
            language="text",
        )

    # Live Visual Inspection for Defense 1
    st.markdown("##### Live Visual Inpainting Preview:")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(attacked_img, width=280, caption=f"1. Attacked Input (with '{adv_text_global}')")
    with preview_col2:
        masked_preview = mask_typographic_region(attacked_img, demo_bbox)
        st.image(masked_preview, width=280, caption="2. Defended Image (Text Region Masked)")

    st.divider()

    # ----------------- DEFENSE 2 -----------------
    st.subheader("🛡️ Defense Mechanism 2: Defense-Prefix Prompting (Prompt Ensembling)")
    st.markdown("**Domain:** Language Conditioning (Text Embedding Space)")

    col_d2_text, col_d2_diag = st.columns([1, 1.2])

    with col_d2_text:
        st.markdown(
            """
            #### How It Works:
            1. **Inductive Constraint Injection:** Instead of passing standard class names (`"a photo of a {CLASS}"`), the prompt is wrapped with an explicit negative instruction.
            2. **Subspace Steering:** The text transformer shifts its output vector away from typographic semantics and projects onto visual/geometry features.
            
            #### Prompt Template:
            ```python
            defended_prompt = (
                "a photo focused solely on the physical object, "
                "ignoring all written text, typography, and stickers of a {CLASS}"
            )
            ```
            
            #### Key Benefit:
            **Zero inference latency overhead** and no external OCR models required. Operates completely through prompt engineering.
            """
        )

    with col_d2_diag:
        st.markdown("**Behind the Scenes Vector Geometry:**")
        st.code(
            f"""
        Typography Subspace
                 ▲
                 │        • Standard Prompt Vector: "{true_label}"
                 │       /  (Cross-contaminated with typography tokens)
                 │      /
                 │     /
                 │    /
                 │   /
                 │  /
                 │ /      • Defended Prompt Vector:
                 │/         "a photo focused solely on physical features..."
  ───────────────┼───────────► Visual & Geometry Subspace
                 │
                 │  (By steering the prompt away from the text axis,
                 │   the dot product with typographic patches collapses!)
            """,
            language="text",
        )

    st.divider()

    # ----------------- ADVANCED EXPANDERS -----------------
    with st.expander("🔬 View Advanced Defenses: Orthogonal Projection (CMOP) & ViT CAP-Pruning"):
        st.markdown(
            """
            ### Defense 3: Subspace Orthogonal Projection (CMOP / LCAR)
            Operates in the multi-modal metric space $\\mathbb{R}^d$. Uses Gram-Schmidt orthogonalization to mathematically strip the adversarial text vector component $\\mathbf{t}_{\\text{distractor}}$ from the visual vector $\\mathbf{v}$:
            """
        )
        st.code(
            """
       v (Contaminated Image Vector)
      /│
     / │
    /  │  v_defended = Normalize(v - alpha * (v · t_distractor) * t_distractor)
   /   ▼  (Subspace projection subtracts the distractor direction)
  ┌─────
  ▼
──┴─────────────────────────────────► t_distractor (Adversarial Text Direction)
            """,
            language="text",
        )

        st.markdown(
            """
            ### Defense 4: ViT Spatial Token Suppression (CAP-Pruning)
            Extracts individual $7 \\times 7 = 49$ patch embeddings from the Vision Transformer. Computes cross-modal attention between each patch and candidate text queries. Patches with an abnormally high text alignment score are masked with a dynamic soft-weight:
            """
        )
        st.code(
            f"""
  ViT 7x7 Patch Grid:
  ┌───┬───┬───┬───┬───┬───┬───┐
  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
  ├───┼───┼───┼───┼───┼───┼───┤
  │ 8 │[{adv_text_global[:1]}]│[{adv_text_global[1:2]}]│[{adv_text_global[2:3]}]│[{adv_text_global[3:4]}]│ 13│ 14│ ◄── High text alignment -> Soft-weight w_i → 0 (Pruned!)
  ├───┼───┼───┼───┼───┼───┼───┤
  │15 │16 │ 🍎│ 🍎│ 🍎│20 │ 21│ ◄── Object patches -> Soft-weight w_i → 1 (Retained)
  └───┴───┴───┴───┴───┴───┴───┘
            """,
            language="text",
        )


# =========================================================================
# TAB 3: INTERACTIVE LIVE COMPARISON TESTBED
# =========================================================================
with tab3:
    st.header("Interactive Live Typographic Vulnerability & Defense Testbed")
    st.markdown("Modify text parameters in real-time and observe how each defense holds up.")

    c_text, c_scale, c_mode = st.columns([2, 2, 2])
    with c_text:
        # Default value bound to the sidebar adversarial text
        text_overlay = st.text_input("Typographic Overlay Text:", value=adv_text_global)
    with c_scale:
        font_scale = st.slider("Font Scale (pt):", min_value=12, max_value=96, value=48, step=4)
    with c_mode:
        defense_mode = st.selectbox(
            "Active Defense Mode:",
            [
                "No Defense (Baseline CLIP)",
                "Visual Defense (Text Masking)",
                "Prompt Defense (Defense-Prefix)",
                "Subspace Defense (Orthogonal Projection)",
                "ViT Defense (CAP-Pruning)",
            ],
        )

    # Dynamically build labels for the current overlay text
    active_labels = [true_label, f"a photo of a {text_overlay.strip().lower()}"]

    # Render attacked image with the active text
    current_attacked, current_bbox = overlay_text_on_image(
        base_image, text=text_overlay, font_size=font_scale
    )

    # Compute baseline
    undefended = get_clip_confidence(processor, model, current_attacked, active_labels)

    # Compute active defense
    if defense_mode == "Visual Defense (Text Masking)":
        display_img = mask_typographic_region(current_attacked, current_bbox)
        defended = get_clip_confidence(processor, model, display_img, active_labels)
        defense_title = "Visual Text Masking"
    elif defense_mode == "Prompt Defense (Defense-Prefix)":
        display_img = current_attacked
        prefix = "a visual photograph focused strictly on the physical object, ignoring all overlaid words, typography, and text stickers of a"
        defended = get_clip_confidence(processor, model, display_img, active_labels, defense_prefix=prefix)
        defense_title = "Defense-Prefix Prompting"
    elif defense_mode == "Subspace Defense (Orthogonal Projection)":
        display_img = current_attacked
        defended = get_clip_confidence_orthogonal_defense(processor, model, display_img, active_labels, distractor_idx=1)
        defense_title = "Orthogonal Subspace Projection"
    elif defense_mode == "ViT Defense (CAP-Pruning)":
        display_img = current_attacked
        defended = get_clip_confidence_cap_pruning(processor, model, display_img, active_labels)
        defense_title = "ViT Patch CAP-Pruning"
    else:
        display_img = current_attacked
        defended = undefended
        defense_title = "None (Standard Baseline)"

    col_view1, col_view2 = st.columns(2)
    with col_view1:
        st.subheader("Base Visual Input")
        st.image(base_image, width=320, caption="Unmodified Ground Truth")
    with col_view2:
        st.subheader(f"Input to CLIP Vision Encoder ({defense_title})")
        st.image(display_img, width=320, caption=f"Overlaid '{text_overlay}' (Font: {font_scale}pt)")

    st.divider()

    # Results & Side-by-Side Comparison
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### ❌ Standard Undefended CLIP")
        for lbl in active_labels:
            val = undefended[lbl]
            st.write(f"**{lbl}**")
            st.progress(float(val))
            st.caption(f"Confidence: `{val:.4f}` ({val*100:.2f}%)")

    with res_col2:
        st.markdown(f"#### 🛡️ Defended CLIP ({defense_title})")
        for lbl in active_labels:
            val = defended[lbl]
            st.write(f"**{lbl}**")
            st.progress(float(val))
            st.caption(f"Confidence: `{val:.4f}` ({val*100:.2f}%)")

    # Bar chart comparison
    df_compare = pd.DataFrame(
        {
            "Label": [l.replace("a photo of a ", "") for l in active_labels],
            "Standard CLIP": [undefended[l] for l in active_labels],
            "Defended CLIP": [defended[l] for l in active_labels],
        }
    ).set_index("Label")

    fig_bar, ax_bar = plt.subplots(figsize=(7.5, 3.5))
    df_compare.plot(kind="bar", ax=ax_bar, color=["#e74c3c", "#2ecc71"], rot=0)
    ax_bar.set_ylabel("Softmax Confidence", fontsize=10)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.7)
    st.pyplot(fig_bar)

    # Detailed Decision Matrix Table
    st.subheader("Detailed Decision Matrix")
    comparison_df = pd.DataFrame(
        {
            "Class Label": active_labels,
            "Standard CLIP Confidence": [f"{undefended[l]*100:.2f}%" for l in active_labels],
            "Defended CLIP Confidence": [f"{defended[l]*100:.2f}%" for l in active_labels],
            "Delta (Defense Impact)": [
                f"{(defended[l] - undefended[l])*100:+.2f}%" for l in active_labels
            ],
        }
    )
    st.dataframe(comparison_df, use_container_width=True)