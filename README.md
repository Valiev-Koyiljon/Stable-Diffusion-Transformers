# 🌌 Stable Diffusion Transformer Experiments

This repository explores the frontiers of image-to-image generation using the Stable Diffusion transformer. Through various experiments, we demonstrate the power and flexibility of text-guided image transformation while maintaining control over the generation process.

## 🎯 Project Overview

This project showcases how to perform text-guided image-to-image generation using the Stable Diffusion model from Hugging Face's Diffusers library. Our experiments demonstrate the model's capability to transform existing images based on text prompts while providing fine-grained control over the transformation process.

## ✨ Features

- Text-guided image-to-image generation with customizable prompts
- Fine-grained control over transformation strength
- GPU acceleration for faster generation
- Reproducible results with seed control
- Support for high-resolution image processing

## 🚀 Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Hugging Face account (for accessing models)

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/Valiev-Koyiljon/Stable-Diffusion-Transformers.git
cd Stable-Diffusion-Transformers
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Login to Hugging Face Hub:
```python
from huggingface_hub import notebook_login
notebook_login()
```

## 💻 Usage

1. Load the pipeline:
```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
```

2. Prepare your input image:
```python
from PIL import Image
import requests
from io import BytesIO

def image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image_rgb = image.convert('RGB')
    img = image_rgb.resize((768, 512))
    return img
```

3. Generate the transformed image:
```python
prompt = "Your text prompt here"
generator = torch.Generator(device=device).manual_seed(42)

transformed_image = pipe(
    prompt=prompt,
    image=image,
    strength=0.75,  # Controls noise level (0.0 to 1.0)
    guidance_scale=7.5,
    generator=generator
).images[0]
```

## 🎮 Parameters

- `strength`: Float between 0.0 and 1.0. Controls how much noise is added to the input image. Higher values allow for more variation but less semantic consistency with the input.
- `guidance_scale`: Float value that guides the strength of the text prompt.
- `generator`: Torch generator for reproducible results.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📚 Resources

- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## ✨ Acknowledgments

- Hugging Face Diffusers library
- CompVis Stable Diffusion model
- The open-source AI community

## 📧 Contact

For any questions or feedback, please feel free to reach out through GitHub issues.

Repository: [Stable-Diffusion-Transformers](https://github.com/Valiev-Koyiljon/Stable-Diffusion-Transformers)
