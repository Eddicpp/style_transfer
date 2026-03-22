# 🎨 Neural Art Studio

**Neural Style Transfer with VGG-19 — fusing photographic content with artistic style through deep feature optimization.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

<p align="center">
  <img width="380" height="200" alt="step_18400" src="https://github.com/user-attachments/assets/f6997ea1-f804-448d-baee-19399222f7c5" />
</p>

<!-- Replace hero_banner.png with a side-by-side: Content | Style | Output -->

---

## 📖 Introduction

Neural Art Studio is a from-scratch implementation of **Neural Style Transfer** based on the seminal work of [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576). The system leverages a pre-trained **VGG-19** network as a perceptual feature extractor, then iteratively optimizes the pixel values of a target image to simultaneously preserve the semantic structure of a content photograph and adopt the visual texture of an artistic reference.

Unlike feed-forward approaches that trade quality for speed, this optimization-based method produces high-fidelity results by directly solving the style transfer objective — making it ideal for research, experimentation, and gallery-quality output.

> **Engineering focus:** Beyond the core algorithm, this project addresses real-world deployment challenges — MPS hardware acceleration on Apple Silicon, gradient stability with in-place operations, and proper ImageNet normalization for perceptually accurate rendering.

---

## 🧮 Mathematical Background

The style transfer objective minimizes a weighted combination of two loss functions over the pixel space of the generated image **x**:

$$\mathcal{L}_{\text{total}}(x) = \alpha \cdot \mathcal{L}_{\text{content}}(x, c) + \beta \cdot \mathcal{L}_{\text{style}}(x, s)$$

where **c** is the content image, **s** is the style image, and α/β control the content-style balance (typically α/β = 1/10⁶).

### Content Loss

Content is preserved by comparing feature activations at a deep convolutional layer, where neurons encode high-level semantic structure rather than raw pixels:

$$\mathcal{L}_{\text{content}} = \frac{1}{2} \sum_{i,j} \left( F^{l}_{ij}(x) - F^{l}_{ij}(c) \right)^2$$

Computed at **`conv4_2`** — deep enough to capture object geometry and spatial arrangement, while discarding surface-level texture that would interfere with stylization.

### Style Loss & the Gram Matrix

Style is captured through correlations between feature maps across multiple layers. The **Gram Matrix** G encodes these correlations, effectively representing the statistical distribution of textures, colors, and patterns at a given network depth:

$$G^{l}_{ij} = \sum_{k} F^{l}_{ik} \cdot F^{l}_{jk}$$

$$\mathcal{L}_{\text{style}} = \sum_{l \in \mathcal{S}} w_l \cdot \frac{1}{4 N_l^2 M_l^2} \sum_{i,j} \left( G^{l}_{ij}(x) - G^{l}_{ij}(s) \right)^2$$

Style loss is aggregated across five layers — **`conv1_1`** through **`conv5_1`** — to capture artistic patterns at every scale: from fine brushstrokes and color palette (shallow layers) to large compositional motifs (deep layers).

### Optimization Target

A critical distinction from standard deep learning: **backpropagation updates the image pixels, not the network weights.** VGG-19 remains frozen as a fixed perceptual loss function while the generated image is iteratively refined via the **Adam** optimizer.

---

## 🏗️ Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │              VGG-19 (Frozen Weights)            │
                    │                                                 │
  Content Image ──► │  conv1_1 ► conv2_1 ► conv3_1 ► conv4_2 ► conv5_1  │
                    │    │         │         │         │         │     │
                    │    ▼         ▼         ▼         ▼         ▼     │
                    │  Style₁   Style₂   Style₃   Content   Style₅   │
                    │    │         │         │     Loss ▲      │     │
                    │    ▼         ▼         ▼         │       ▼     │
                    │    └────── Gram Matrices ────────┼───────┘     │
                    │              │                   │              │
                    │              ▼                   │              │
                    │         Style Loss               │              │
                    └─────────────┼───────────────────┼──────────────┘
                                  │                   │
                                  ▼                   ▼
                         ┌──────────────────────────────────┐
                         │   𝓛_total = α·𝓛_content + β·𝓛_style   │
                         └──────────────────┬───────────────┘
                                            │
                                     Adam Optimizer
                                            │
                                            ▼
                                   Generated Image (x)
                                  ◄── pixel updates ──►
```

---

## ⚡ Installation & Usage

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/your-username/neural-art-studio.git
cd neural-art-studio

# Install dependencies
pip install torch torchvision matplotlib Pillow numpy
```

### Quick Start

```python
import torch
from style_transfer import NeuralStyleTransfer

# Automatic device selection (MPS → CUDA → CPU)
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

nst = NeuralStyleTransfer(device=device)

output = nst.transfer(
    content_path="images/content/photograph.jpg",
    style_path="images/style/starry_night.jpg",
    steps=1000,
    alpha=1,           # Content weight
    beta=1e6,          # Style weight
    learning_rate=0.003,
    show_every=200,    # Live preview interval
)

output.save("results/output.jpg")
```

### Command Line

```bash
python run.py \
    --content images/content/photograph.jpg \
    --style images/style/starry_night.jpg \
    --steps 1500 \
    --alpha 1 \
    --beta 1e6 \
    --device mps \
    --output results/output.jpg
```

---

## 🍎 Optimization on macOS (MPS)

Running deep optimization loops on Apple Silicon required solving several non-trivial engineering problems.

### Problem 1: In-Place ReLU Gradient Corruption

The pre-trained VGG-19 from `torchvision` uses `inplace=True` on all ReLU layers. This is fine for standard inference, but in our pipeline — where gradients must flow *backward through the frozen network* to update the input image — in-place operations **corrupt the computation graph** and cause silent NaN propagation or outright crashes.

**Solution:** Systematically replace every in-place ReLU before the forward pass:

```python
for i, layer in enumerate(model.features):
    if isinstance(layer, torch.nn.ReLU):
        model.features[i] = torch.nn.ReLU(inplace=False)
```

This preserves intermediate activations needed for correct gradient computation at a modest memory cost — a worthwhile trade-off for numerical stability.

### Problem 2: ImageNet Normalization Mismatch

VGG-19 expects inputs normalized with ImageNet statistics. Without proper normalization, feature activations are off-distribution and the loss landscape becomes erratic. Conversely, the generated image must be **denormalized** before display or saving.

```python
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).to(device)

def normalize(img):
    return (img - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]

def denormalize(img):
    return img * IMAGENET_STD[None, :, None, None] + IMAGENET_MEAN[None, :, None, None]
```

### Problem 3: MPS Memory Pressure

Large images (1024px+) on 8GB unified memory machines can trigger out-of-memory errors during the backward pass. The pipeline includes automatic resolution scaling and explicit `torch.mps.empty_cache()` calls between optimization steps when memory pressure is detected.

---

## 🖼️ Visual Results

### Example 1 — Van Gogh's cat

| Content | Style | Output |
|:--:|:--:|:--:|
| ![content](https://github.com/user-attachments/assets/78aa32d5-4c67-4668-b256-02b8c20799b8)
) | ![style](https://github.com/user-attachments/assets/e84c715b-0fc6-45fc-b7c1-d8ce668525f3)
 | ![Output](<img width="380" height="200" alt="image" src="https://github.com/user-attachments/assets/32e41f81-d0ea-4db9-aed5-272d40bfceca" />) |
| *Countryside photograph* | *Monet — Water Lilies* | *1000 steps · α=1 · β=10⁶* |

### Optimization Progress

<p align="center">
  <img src="assets/results/optimization_progress.gif" alt="Optimization over 1000 steps" width="600"/>
</p>

<!-- Tip: generate this GIF by saving frames at show_every intervals and stitching with ffmpeg or imageio -->

---

## 🎛️ Hyperparameter Guide

| Parameter | Range | Effect |
|:--|:--|:--|
| **α / β ratio** | 1/10⁵ — 1/10⁷ | Lower ratio → stronger stylization; higher → more photographic fidelity |
| **Steps** | 500 — 2000 | More steps → finer convergence. Diminishing returns beyond ~1500 for most pairs |
| **Learning Rate** | 0.001 — 0.01 | Higher LR converges faster but risks instability; 0.003 is a reliable default |
| **Image Size** | 256 — 1024px | Larger → richer detail but quadratic memory/time cost |
| **Style Layers** | `conv1_1` — `conv5_1` | Subset selection biases toward fine (shallow) or coarse (deep) texture transfer |

> **Practical tip:** Start with 500 steps at 512px to quickly evaluate a content-style pairing, then scale up resolution and step count for your final render.

---

## 📂 Repository Structure

```
neural-art-studio/
├── images/
│   ├── content/              # Input photographs
│   └── style/                # Reference artworks
├── outputs/
├── utils.py
├── vgg_model.py
├── main.py                    # CLI entry point
└── README.md
```

---

## 🗺️ Roadmap

- [x] Core NST pipeline (Gatys et al. optimization-based approach)
- [x] MPS acceleration for Apple Silicon
- [x] ReLU in-place stability fix
- [x] Live Matplotlib preview during optimization
- [ ] **Semantic Style Transfer** — integrate **YOLOv8-Segmentation** to apply style selectively (e.g., stylize only the subject while preserving a photorealistic background, or vice versa)
- [ ] Feed-forward approximation (Johnson et al.) for real-time inference
- [ ] Multi-style interpolation (blend multiple artworks with weighted Gram matrices)
- [ ] Gradio / Streamlit web interface
- [ ] ONNX export for cross-platform deployment

---

## 📚 References

1. **Gatys, L. A., Ecker, A. S., & Bethge, M.** (2015). *A Neural Algorithm of Artistic Style.* [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
2. **Simonyan, K. & Zisserman, A.** (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
3. **Johnson, J., Alahi, A., & Fei-Fei, L.** (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)

---

## 🤝 Contributing

Contributions, ideas, and style experiments are welcome. If you produce an interesting content-style combination or discover a useful hyperparameter regime, consider opening a PR to add it to the gallery.

---

<p align="center">
  <em>Where the gradient meets the brushstroke.</em>
</p>
