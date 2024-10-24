import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import argparse
import torch
import cv2
import numpy as np
from diffusers import LCMScheduler, AutoPipelineForText2Image


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "/data/AI_DATA/share/modelscope/hub/VAST-AI-Research/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)




# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()

# t2i model 
model_id = "AI-ModelScope/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download(model_id)
sd_pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16, variant="fp16")
sd_pipe.enable_model_cpu_offload()
sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe.to("cuda")
# load and fuse lcm lora
sd_pipe.load_lora_weights(adapter_id)
sd_pipe.fuse_lora()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def generate_image(prompt, negtivate_prompt, guidance_scale=1, num_inference_steps=10):
    image = sd_pipe(prompt=prompt, negtivate_prompt = negtivate_prompt,  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=1024, width=1024, generator=torch.Generator("cpu").manual_seed(0)).images[0]
    # 将 PIL 图像转换为 NumPy 数组
    numpy_array = np.array(image)
    # 使用 OpenCV 从 NumPy 数组创建 cv::Mat 对象
    image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2RGBA)
    return image
def generate(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = generate(preprocessed, 256, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # TripoSR Demo
    [TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).
    
    **Tips:**
    1. If you find the result is unsatisfied, please try to change the foreground ratio. It might improve the results.
    2. It's better to disable "Remove Background" for the provided examples (except fot the last one) since they have been already preprocessed.
    3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
    """
    )
    with gr.Row():
        common_negtivate_prompt = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
        prompt_input = gr.Textbox(label="image prompt", value="a cartoonish teddy bear picture with a white background, enhanced with 3D effects to give it a playful, three-dimensional appearance. The focus should be on capturing the essence of a cuddly teddy bear in a vivid and imaginative way, suitable for a children's book illustration or a fun, whimsical wallpaper")
        neg_prompt_input = gr.Textbox(label="negtivate prompt", value=common_negtivate_prompt)
        guidance_scale = gr.Number(label="guidance scale", value=1.0, minimum=0, maximum=10)
        num_steps = gr.Number(label="inference steps", value=10, minimum=4, maximum=20)
        sd_button = gr.Button("生成图像")
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Output Model (GLB Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/hamburger.png",
                "examples/poly_fox.png",
                "examples/robot.png",
                "examples/teapot.png",
                "examples/tiger_girl.png",
                "examples/horse.png",
                "examples/flamingo.png",
                "examples/unicorn.png",
                "examples/chair.png",
                "examples/iso_house.png",
                "examples/marble.png",
                "examples/police_woman.png",
                "examples/captured.jpeg",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb],
            cache_examples=False,
            fn=partial(run_example),
            label="Examples",
            examples_per_page=20,
        )
    sd_button.click(generate_image, [prompt_input, neg_prompt_input,  guidance_scale, num_steps], input_image)
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution],
        outputs=[output_model_obj, output_model_glb],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None, 
        server_port=args.port
    )