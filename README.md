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

## Examples
<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Output Video</b></td>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>Output Video</b></td>
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A Porsche car is moving on the desert"</td>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A jeep car is moving on the snow"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/jeep-moving_Porsche.gif"></td>
  <td style colspan="2"><img src="examples/jeep-moving_snow.gif"></td>       
</tr>


<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is running"</td>
  <td width=25% style="text-align:center;">"Stephen Curry is running in Time Square"</td>
  <td width=25% style="text-align:center;color:gray;">"A man is running"</td>
  <td width=25% style="text-align:center;">"A man is running in New York City"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/man-running_stephen.gif"></td>
  <td style colspan="2"><img src="examples/man-running_newyork.gif"></td>       
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A child is riding a bike on the road"</td>
  <td width=25% style="text-align:center;">"a child is riding a bike on the flooded road"</td>
  <td width=25% style="text-align:center;color:gray;">"A child is riding a bike on the road"</td>
  <td width=25% style="text-align:center;">"a lego child is riding a bike on the road.gif"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/child-riding_flooded.gif"></td>
  <td style colspan="2"><img src="examples/child-riding_lego.gif"></td>       
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A car is moving on the snow"</td>
  <td width=25% style="text-align:center;color:gray;">"A car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A jeep car is moving on the desert"</td>
</tr>

<tr>
  <td style colspan="2"><img src="examples/red-moving_snow.gif"></td>
  <td style colspan="2"><img src="examples/red-moving_desert.gif"></td>       
</tr>
</table>

## citation

If you make use of our work, please cite our paper.

```
@article{swin-editor,
  title={Swin-Editor: A Comprehensive Framework for Text-Driven Video Editing},
  author={Author1, Author2, Author3},
  journal={WACV 2025},
  year={2024}
}
```
