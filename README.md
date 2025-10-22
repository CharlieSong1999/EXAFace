# Extreme amodal face detection

This repository contains the implementation of the paper "Extreme amodal face detection"

<!-- Keywords: Image Inpainting, Diffusion Models, Image Generation -->

> [Changlin Song](https://charliesong1999.github.io/)<sup>1</sup>, [Yunzhong Hou](https://hou-yz.github.io/)<sup>1</sup>, [Michael Randall Barnes](https://xinntao.github.io/)<sup>2</sup>, [Rahul Shome](https://rahulsho.me/)<sup>1</sup>, [Dylan Campbell](https://sites.google.com/view/djcampbell)<sup>1</sup><br>
> <sup>1</sup>Australian National University <sup>2</sup>University of Oslo


<p align="center">
  <a href="https://charliesong1999.github.io/exaft_web/">🌐Project Page</a> |
  <a href="https://arxiv.org/abs/2510.06791">📜Arxiv</a> |
  <a href="https://drive.google.com/drive/folders/1FM-YG7vuBazMkVu-3es4PPXXuxigHiuF?dmr=1&ec=wgc-drive-hero-gotof">🗄️Data</a>
  <!-- <a href="https://drive.google.com/file/d/1IkEBWcd2Fui2WHcckap4QFPcCI0gkHBh/view">📹Video</a> |
  <a href="https://huggingface.co/spaces/TencentARC/BrushNet">🤗Hugging Face Demo</a> | -->
</p>

Extreme amodal detection is the task of inferring the 2D location of objects that are not fully visible in the input image but are visible within an expanded field-of-view. This differs from amodal detection, where the object is partially visible within the input image, but is occluded. In this paper, we consider the sub-problem of face detection, since this class provides motivating applications involving safety and privacy, but do not tailor our method specifically to this class. Existing approaches rely on image sequences so that missing detections may be interpolated from surrounding frames or make use of generative models to sample possible completions. In contrast, we consider the single-image task and propose a more efficient, sample-free approach that makes use of the contextual cues from the image to infer the presence of unseen faces. We design a heatmap-based extreme amodal object detector that addresses the problem of efficiently predicting a lot (the out-of-frame region) from a little (the image) with a selective coarse-to-fine decoder. Our method establishes strong results for this new task, even outperforming less efficient generative approaches.

# Train/eval/inferece

Go to sc2f/

# Comparsion methods

Coming soon...


