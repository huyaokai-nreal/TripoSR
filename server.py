from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
import numpy as np
import cv2
from PIL import Image
from ttsr import TTSR
import base64

app = FastAPI()

# Initialize the TTSR model
ttsr = TTSR(device="cuda")
ttsr.init_model()

class TextTo3DRequest(BaseModel):
    prompt: str
    negative_prompt: str
    guidance_scale: float = 1
    num_inference_steps: int = 10
    remove_background: bool = True
    foreground_ratio: float = 0.85
    mc_resolution: int = 320
    output_format: str = "obj"

@app.post("/text-to-3d/")
async def text_to_3d(request: TextTo3DRequest):
    _, _, result_path = ttsr.txt_to_3d(
        request.prompt,
        request.negative_prompt,
        request.guidance_scale,
        request.num_inference_steps,
        request.remove_background,
        request.foreground_ratio,
        request.mc_resolution,
        [request.output_format]
    )
    with open(result_path[0], 'rb') as f:
        obj_content = f.read()
    
    # Encode the content to base64
    obj_content_base64 = base64.b64encode(obj_content).decode('utf-8')
    
    return {
        "result_path": result_path,
        "obj_content": obj_content_base64
    }