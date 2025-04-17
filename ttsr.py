import tempfile
import numpy as np
import rembg
import torch
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from transformers import MarianTokenizer, MarianMTModel
import torch
import cv2
import numpy as np
from diffusers import LCMScheduler, AutoPipelineForText2Image
class TTSR:
    def __init__(self, device):
        self.device = device
        self.tsr_model_path = "./models/modelscope/hub/VAST-AI-Research/TripoSR/"
        self.sdxl_model_path = "./models/modelscope/hub/AI-ModelScope/stable-diffusion-xl-base-1___0"
        # 新增翻译模型初始化
        self.translation_model_name = "./models/modelscope/hub/opus-mt-zh-en"
        self.tokenizer = MarianTokenizer.from_pretrained(self.translation_model_name)
        self.translation_model = MarianMTModel.from_pretrained(self.translation_model_name).to(device)
        self.adapter_id = "latent-consistency/lcm-lora-sdxl"
        self.common_prompt = "3D cartoon, white background"
        self.common_negative_prompt = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"

    def init_model(self):
        self.tsr_model = TSR.from_pretrained(self.tsr_model_path, config_name="config.yaml", weight_name="model.ckpt")
        self.tsr_model.renderer.set_chunk_size(8192)
        self.tsr_model.to(self.device)
        self.rembg_session = rembg.new_session() 
        self.sdxl_model = AutoPipelineForText2Image.from_pretrained(self.sdxl_model_path, torch_dtype=torch.float16, variant="fp16")
        self.sdxl_model.enable_model_cpu_offload()
        self.sdxl_model.scheduler = LCMScheduler.from_config(self.sdxl_model.scheduler.config)
        self.sdxl_model.to("cuda")
        # load and fuse lcm lora
        self.sdxl_model.load_lora_weights(self.adapter_id)
        self.sdxl_model.fuse_lora()
        self.translation_model.eval()  # 翻译模型设置为评估模式

    def preprocess(self, input_image, do_remove_background, foreground_ratio):
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            return image

        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, self.rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
        return image

    def generate_image(self, prompt, negtivate_prompt, guidance_scale=1, num_inference_steps=10):
        # 新增：先翻译中文提示词
        print(prompt)
        if not self.is_pure_english(prompt):
            if any([u'\u4e00' <= char <= u'\u9fff' for char in prompt]):  # 检测中文
                prompt = self._translate_to_english(prompt)
            if any([u'\u4e00' <= char <= u'\u9fff' for char in negtivate_prompt]):
                negtivate_prompt = self._translate_to_english(negtivate_prompt)
        english_prompt = prompt
        # 原有提示词拼接逻辑
        prompt = f"{prompt}, {self.common_prompt}"
        negtivate_prompt = f"{negtivate_prompt}, {self.common_negative_prompt}"
        #generator=torch.Generator("cpu").manual_seed(0)
        image = self.sdxl_model(prompt=prompt, negtivate_prompt = negtivate_prompt,  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=1024, width=1024).images[0]
        numpy_array = np.array(image)
        image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2RGBA)
        return image, english_prompt

    def image_to_3d(self, image, mc_resolution, formats=["obj", "glb"]):
        scene_codes = self.tsr_model(image, device=self.device)
        mesh = self.tsr_model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
        mesh = to_gradio_3d_orientation(mesh)
        rv = []
        for format in formats:
            mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
            mesh.export(mesh_path.name)
            rv.append(mesh_path.name)
        return rv
    
    def txt_to_3d(self,  prompt, negtivate_prompt, guidance_scale=1, num_inference_steps=10, rm_bg=True, fg_ration=0.85, mc_resolution=320, output_formats=["obj"]):
        raw_image,_ = self.generate_image(prompt, negtivate_prompt, guidance_scale, num_inference_steps)
        processed_image = self.preprocess(Image.fromarray(raw_image), rm_bg, fg_ration)
        result_path = self.image_to_3d(processed_image, mc_resolution, output_formats)
        return raw_image, processed_image, result_path

    def _translate_to_english(self, text: str) -> str:
        """将中文文本翻译成英文"""
        if not text.strip():  # 处理空输入
            return ""
        
        # 处理中文标点转英文（可选，提升提示词质量）
        text = text.replace("，", ",").replace("。", ".").replace("！", "!").replace("？", "?")
        
        # 翻译逻辑
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.translation_model.generate(**inputs, max_length=512)
        translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return translated_text

    @staticmethod
    def is_pure_english(text):
        """
        判断文本是否为纯英文（含字母、数字、空格、常见标点）
        """
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()_+{}|:\"<>?,.;'[]\\-=")
        for char in text:
            if char not in allowed_chars:
                return False
        return True
