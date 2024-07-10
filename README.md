<div align="center">

<h1>Swin-Editor: A Comprehensive Framework for Text-Driven Video Editing</h1>

<br>
<image src="docs/Overview (4)-1.png" />
<br>

</div>

Large visual models have recently made considerable progress in Text-to-Video generation thanks to the development of foundation models and multi-modal alignment techniques, making video generation more and more realistic. Current approaches predominantly rely on adapting image-based diffusion models via spatio-temporal attention, but this generally leads to temporal inconsistency and increasing model complexity. This inconsistency is mainly related to the fact those approaches are founded on models that were originally designed for image generation, thus, they do not consider implicitly the spatio-temporal aspect
of videos. In this article, we introduce Swin-Editor, an efficient approach of video editing from text-instruction that expands a diffusion-based Text-to-Image model into Text-to-Video. Specifically, our focus lies in enhancing the visual quality of the generated videos by incorporating a spatio-temporally factorized video prediction mechanism in the diffusion model. Additionally, to reduce computational complexity and memory requirements, the proposed model includes a Vector Quantized Variational Autoencoders module, intended to quantize and compress the spatio-temporal latent features. The proposed architecture produces a good compromise between multiple evaluation metrics against state-of-the-art models in various scenarios.

## Installation
### Requirements

```shell
pip install -r requirements.txt
```
Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for improved efficiency and speed on GPUs. 

## Run Demo

```bash
accelerate launch test_swineditor.py --config path/to/config
```

