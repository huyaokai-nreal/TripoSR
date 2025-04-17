import requests
import base64

url = "http://10.2.6.25:8000/text-to-3d/"
data = {
    "prompt": "一个red and big 苹果",
    "negative_prompt": "ugly, deformed",
    "guidance_scale": 1.0,
    "num_inference_steps": 10,
    "remove_background": True,
    "foreground_ratio": 0.85,
    "mc_resolution": 320,
    "output_format": "obj"
}

response = requests.post(url, json=data)
print(response.json()['result_path'])
obj_data = response.json()['obj_content']
obj_content = base64.b64decode(obj_data)
# 保存为文件
with open("output.obj", "wb") as f:
    f.write(obj_content)
