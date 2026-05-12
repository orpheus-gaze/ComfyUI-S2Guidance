# ComfyUI-S2Guidance 🚀

A lightweight ComfyUI extension implementing **S²-Guidance** and **Perpo-Guidance**, two advanced sampling techniques designed to improve prompt adherence and generation robustness compared to standard CFG. **S²-Guidance** is based on the methodology described in [arXiv:2508.12880](https://arxiv.org/abs/2508.12880), while **Perpo-Guidance** is my personal take on it with slightly different operations.

## 📦 Installation
1. Navigate to your `ComfyUI/custom_nodes/` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/orpheus-gaze/ComfyUI-S2Guidance.git
   ```
3. No extra pip dependencies are required (uses standard `torch`, `numpy`, and ComfyUI's core APIs).
4. Restart ComfyUI or refresh your browser window.

## 🎛️ Usage & Nodes
After installation, two new nodes will appear in your node search under the **✨ S2GUIDANCE** category:
- `✨S2GuidanceDIT` (S²-Guidance)
- `✨Perpo-GuidanceDIT` (Perpo-Guidance)

Simply insert either node **between your base model and your sampler/denoise node**, then wire the output to your usual workflow. Adjust the parameters below to fine-tune results.

## ⚙️ Node Parameters
Both nodes share an identical parameter layout:

| Parameter | Type | Default Range | Description |
|-----------|------|---------------|-------------|
| `guidance_scale` | Float | `0.0 – 2.0` (S²) / `0.0 – 10.0` (Perpo) | Strength of the guidance effect. `0.0` = disabled, higher values = stronger impact on structure/detail. |
| `skip_layers_percentage` | Int | `1 – 100` (Default: `1`) | Percentage of model layers dynamically skipped during inference to compute guidance predictions. |

## 🧠 How It Works
- **S²-Guidance:** Subtracts a weighted subnetwork prediction from the standard CFG output, then applies *energy normalization* to preserve color balance and intensity. Ideal for most modern DIT models. Start with low values for the guidance scale like `0.1-0.5` and note that the `skip_layers_percentage` variable may impact guidance scales differently.
- **Perpo-Guidance:** Instead of normalizing energy, it computes the **orthogonal (perpendicular) component** relative to the CFG vector and subtracts that. Often yields sharper fine details or different stylistic contrasts. Test in the `0.5–2.0` range first.

> 💡 *Both methods dynamically sample which layers to skip each step, making them more robust than static CFG scaling.*

## ✅ Supported Architectures
The extension automatically tries to detect model architecture and applies guidance based on model configuration:
- **Lumina / Z-Image** type models (single-block architecture)
- **Flux 2 / Klein** type models (double + single block architecture)
- **AuraFlow** type models (custom layer configuration)

Unsupported architectures will fall back to standard CFG behavior without errors.
Please note that not all DIT model architectures have been tested, but functionality has been validated on at least Flux Klein, Z-Image, .

## 💡 Usage Tips
- Start with low guidance scales (`0.1–0.5` for S², `0.5–2.0` for Perpo). High values can oversharpen or introduce artifacts.
- `skip_layers_percentage` around `1-10` usually provides the best balance between stability and improvement.
- If a model looks "flat" or desaturated, try switching to Perpo-Guidance or lowering the guidance scale slightly.

## 🕰️ Future
I think there are some possible steps that I might take to improve on the functionality of these nodes. These include:
- Quantitative validation through CLIP and other types of scores to discover optimal parameters
- Biased selection of layers to skip, e.g.: 
    - Skip different layers at different sigma noise levels
    - Focus on layers with more influence on the diffusion process (i.e. double layers)

## 📖 Credits & References
- Research Paper: ["STOCHASTIC SELF-GUIDANCE FOR TRAINING-FREE ENHANCEMENT OF DIFFUSION MODELS" on arXiv (2508.12880)](https://arxiv.org/abs/2508.12880)
- Research Github link: [https://github.com/AMAP-ML/S2-Guidance](https://github.com/AMAP-ML/S2-Guidance)
- Built using ComfyUI's current `comfy_api` extension framework and with inspiration from the SkipLayerGuidance node for layer skipping
- Compatible with any workflow that uses standard DIT model → sampler pipelines
- Many thanks to the cooperation of some friendly LLMs as well

---
*Found a bug or have a suggestion? Open an issue or submit a PR!* 🛠️
