# EchoVideo: Identity-Preserving Human Video Generation by Multimodal Feature Fusion

This repo contains PyTorch model definitions, pre-trained weights and inference code for our video generation model, EchoVideo.
> [**EchoVideo: Identity-Preserving Human Video Generation by Multimodal Feature Fusion**](https://arxiv.org/abs/2501.13452) <be>

# News

**[2025.03.02]** We release the inference code and model weights of EchoVideo. [DownLoad](ckpts/README.md)

# Introduction

EchoVideo is capable of generating a personalized video from a single photo and a text description. It excels in addressing issues related to "semantic conflict" and "copy-paste" problems. And demonstrates state-of-the-art performance.


# Gallery
## 1. Text-to-Video Generation
| Face-ID Preserving | Full-Body Preserving|
| ---- | ---- |
| <img height="240" src="asset/examples/3.gif" > | <img height="240" src="asset/examples/4.gif" > |

## 2. Comparisons
| EchoVideo | ConsisID | IDAnimator |
| ---- | ---- | ---- |
| <img height="180" src="asset/examples/2.gif" > | <img height="180" src="asset/examples/5.gif" > | <img height="180" src="asset/examples/6.gif" > |
| <img height="180" src="asset/examples/1.gif" > | <img height="180" src="asset/examples/7.gif" > | <img height="180" src="asset/examples/8.gif" > |


# Usage
**Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12. Support both gpu and npu**

## Clone the repository:
```shell
git clone https://github.com/bytedance/EchoVideo
cd EchoVideo
```

## Installation
```shell
pip install -r requirements.txt
```
## Download Pretrained Weights
The details of download pretrained models are shown [here](ckpts/README.md).
## Run Demo
```shell
# multi-resolution video generation [(480, 640), (480, 848), (480, 480), (848, 480), (640, 480)]
python infer.py
```

# Methods
## **Overall Architecture**
<p align="center">
  <img src="asset/examples/framework.jpg"  height=350>
</p>

Overall architecture of EchoVideo. By employing a meticulously designed IITF module and mitigating the over-reliance on input images, our model effectively unifies the semantic information between the input facial image and the textual prompt. This integration enables the generation of consistent characters with multi-view facial coherence, ensuring that the synthesized outputs maintain both visual and semantic fidelity across diverse perspectives.

## **Key Features**
<p align="center">
  <img src="asset/examples/IITF.jpg"  height=350>
</p>


Illustration of facial information injection methods. (a) IITF. Facial and textual information are fused to ensure consistent guidance throughout the generation process. we propose IITF to fuse text and facial information, establishing a semantic bridge between facial and textual information, coordinating the influence of different information on character features, thereby ensuring the consistency of generated characters. IITF consists of two core components: facial feature alignment and conditional feature alignment. (b) Dual branch. Facial and textual information are independently injected through Cross Attention mechanisms, providing separate guidance for the generation process.  

## Benchmark

| Model | Identity Average↑ | Identity Variation↓ | Inception Distance↓ | Dynamic Degree↑ |
| -- | -- | -- | -- |-----------------|
| IDAnimator | 0.349 | **0.032** | **159.11** | 0.280           |
| ConsisID | <u>0.414</u> | 0.094 | 200.40 | 0.871           |
| pika | 0.329 | 0.091 | 268.35 | <u>0.954</u>    |
| Ours | **0.516** | <u>0.075</u> | <u>176.53</u> | **0.955**       |

# Acknowledgements
* [CogVideo](https://huggingface.co/THUDM/CogVideoX-5b): The DiT module we adpated from, and the VAE module we used.
* [SigLip](https://huggingface.co/google/siglip-base-patch16-224): Vision Encoder we used.


# BibTeX
If you find our work useful in your research, please consider citing the paper
```bibtex
@article{wei2025echovideo,
  title={EchoVideo: Identity-Preserving Human Video Generation by Multimodal Feature Fusion},
  author={Wei, Jiangchuan and Yan, Shiyue and Lin, Wenfeng and Liu, Boyuan and Chen, Renjie and Guo, Mingyu},
  journal={arXiv preprint arXiv:2501.13452},
  year={2025}
}
```
