import gradio as gr
from functools import partial
import argparse
from ttsr import TTSR 

ttsr_model = TTSR("cuda")
ttsr_model.init_model()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def run_example(image_pil):
    preprocessed = ttsr_model.preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = ttsr_model.generate(preprocessed, 256, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


with gr.Blocks(title="TTSR") as interface:
    with gr.Row():
        prompt_input = gr.Textbox(label="image prompt", value="a  teddy bear")
        neg_prompt_input = gr.Textbox(label="negtivate prompt")
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
    sd_button.click(ttsr_model.generate_image, [prompt_input, neg_prompt_input,  guidance_scale, num_steps], input_image)
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=ttsr_model.preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=ttsr_model.image_to_3d,
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