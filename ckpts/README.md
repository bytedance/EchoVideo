# Download Pretrained Models

All models are stored in `EchoVideo/ckpts`, and the file structure is as follows
```shell
EchoVideo/ckpts/
├── clip_visual_encoder
├── configuration.json
├── face_encoder
├── model_index.json
├── README.md
├── scheduler
│   └── scheduler_config.json
├── text_encoder
├── tokenizer
├── transformer
    ├── config.json
    ├── diffusion_pytorch_model-00001-of-00002.safetensors
    ├── diffusion_pytorch_model-00002-of-00002.safetensors
    └── diffusion_pytorch_model.safetensors.index.json
└── vae
```

## Download EchoVideo Model
To download the EchoVideo model, first install the [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli).

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'EchoVideo'
cd EchoVideo
huggingface-cli download bytedance-research/EchoVideo --local-dir ./ckpts
```

## Download Face Encoder

- Download [Face Recognition](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) model and put them into ckpts/face_encoder/ folder.

```shell
mkdir -p ckpts/face_encoder
cd ckpts/face_encoder
unzip antelopev2.zip -d models
```

2. Face Parse (face_encoder folder), we use the [bisenet](https://huggingface.co/BestWishYsh/ConsisID-preview/resolve/main/face_encoder/parsing_bisenet.pth) model.

```shell
cd ckpts/face_encoder
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
wget https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth

```

## Download Visual Encoder

Visual encoder (clip_visual_encoder folder), we use the [SigLip](https://huggingface.co/google/siglip-base-patch16-224) model.

```shell
cd ckpts
huggingface-cli download google/siglip-base-patch16-224 --local-dir ./clip_visual_encoder
```

## DownLoad Text Encoder, Tokenizer, VAE

Text encoder (text_encoder folder), Tokenizer (tokenizer folder), VAE (vae folder), we follow the [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b)

- Download [text encoder](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/text_encoder) and put them into ckpts/text_encoder folder.
- Download [tokenizer](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/tokenizer) and put them into ckpts/tokenizer folder.
- Download [vae](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/vae) and put them into ckpts/vae folder.
