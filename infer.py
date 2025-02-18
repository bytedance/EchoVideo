# Copyright 2025 Bytedance Ltd. and/or its affiliates
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import argparse
from typing import List, Dict

# Check for specific device type and import necessary modules
if "Ascend910B" in os.environ.get("ARNOLD_DEVICE_TYPE"):
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False

# Importing necessary modules from diffusers and other libraries
from transformers import AutoImageProcessor
from diffusers.utils import export_to_video
from models.pipeline_echovideo import EchoVideoPipeline
from models.siglip_vision_encoder import SiglipForImageClassification
from models.utils import extract_face


def generate_video_from_list(
    model_path: str,
    generate_params: List[Dict[str, str]],
    dtype: torch.dtype = torch.bfloat16,
):
    """Generate video from a list of parameters."""
    
    # Load the EchoVideo pipeline
    pipe = EchoVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe.to("cuda")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    clip_path = os.path.join(model_path, "clip_visual_encoder")
    pipe.image_processor = AutoImageProcessor.from_pretrained(clip_path)
    pipe.model = SiglipForImageClassification.from_pretrained(clip_path, torch_dtype=torch.bfloat16).eval()
    pipe.model.requires_grad_(False)
    pipe.model.to("cuda")

    # Generate the video frames based on the prompt
    for p in generate_params:
        output_path = p["output_path"]
        if os.path.exists(output_path):
            print(f"{output_path} exists")
            continue
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        prompt, image_label = p["prompt"], p["img_path"]
        
        # Extract face from the image
        image, id_cond = extract_face(image_label, model_path, "cuda", torch.float16)
        
        # Generate video using the pipeline
        video_generate = pipe(
            image=image,
            prompt=prompt,
            width=p["width"], 
            height=p["height"],
            num_videos_per_prompt=1,
            num_inference_steps=p["num_inference_steps"],
            num_frames=p["num_frames"],
            use_dynamic_cfg=True,
            guidance_scale=p["guidance_scale"],
            generator=torch.Generator().manual_seed(p["seed"]),
            use_face_embedding=True,
            encoder_face_feats_hidden_states=id_cond,
        ).frames[0]
        
        # Export the generated video
        export_to_video(video_generate, output_path, fps=p["fps"])


def main():
    """Main function to parse arguments and generate video."""
    parser = argparse.ArgumentParser(description="Generate videos from prompts using EchoVideo.")
    parser.add_argument('--model_path', type=str, 
                        default="ckpts", 
                        help='Path to the pre-trained model.')
    parser.add_argument('--prompt', type=str, 
                        default="On a busy city street, a man stood in front of a traffic light, wearing a dark blue windbreaker and a pair of worn black leather shoes. He held a cup of steaming coffee in his hand and a faint smile on his face, as if he was enjoying the short wait. Pedestrians hurried past and the traffic was endless, but he seemed calm and unhurried, looking through the crowd, as if looking for something.", 
                        help='Text prompt for video generation.')
    parser.add_argument('--img_path', type=str, 
                        default="asset/examples/test.png",
                        help='Path to the input image.')
    parser.add_argument('--guidance_scale', type=float, default=6.0, help='Guidance scale for generation.')
    parser.add_argument('--width', type=int, default=848, help='Width of the generated video.')
    parser.add_argument('--height', type=int, default=480, help='Height of the generated video.')
    parser.add_argument('--fps', type=int, default=16, help='Frames per second for the output video.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation.')
    parser.add_argument('--output_path', type=str, 
                        default="results/0000_00011_42.mp4", 
                        help='Output path for the generated video.')

    args = parser.parse_args()

    generate_params = [{
        'prompt': args.prompt,
        'img_path': args.img_path,
        'num_inference_steps': 50,
        'guidance_scale': args.guidance_scale,
        'width': args.width,
        'height': args.height,
        'num_frames': 49,
        'fps': args.fps,
        'seed': args.seed,
        'output_path': args.output_path
    }]

    generate_video_from_list(
        generate_params=generate_params, 
        model_path=args.model_path, 
        dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()