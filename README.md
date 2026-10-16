# CLIP Typographic Attack & Defense Suite

A complete framework to demonstrate, benchmark, and mitigate **typographic attacks** against vision-language models (e.g., OpenAI's CLIP ViT-B/32).

---

##  Problem Overview

Multimodal models trained on web-scale text-image pairs develop **typographic bias**: when an image of an object (e.g., an apple) has conflicting text rendered onto it (e.g., `"Orange"`), the vision encoder prioritizes optical text tokens over visual semantics. 

This repository provides an end-to-end framework to:
1. Replicate typographic vulnerability under variable text scales.
2. Evaluate 4 defense mechanisms (visual masking, prompt prefixes, orthogonal projection, and ViT CAP-pruning).
3. Test and interact with the model via a live Streamlit dashboard.

---

##  Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd clip-typographic-defense
pip install -r requirements.txt
---

### Verification

To verify everything is working:

1. **Run the test suite:**
   ```bash
   pytest test.py -v
streamlit run app.py
