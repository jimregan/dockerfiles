import os
import sys
import json
import random
import ast
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lego.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN
from lego.conversation import SeparatorStyle
from lego import conversation as conversation_lib
from lego.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, load_image_square, postprocess_output
from lego.model.builder import CONFIG, load_pretrained_model

# ============ CONFIG ============

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
model_path = "./ckpt"
# JUST TAKE FIRST FRAME IN EACH _0 
object_json_path = "/shared/mm_conv/screenshots_1.6/hsi_7_0719_227_003_main_0/json_objectID/0000_seg.json"
mask_img_path = "/shared/mm_conv/screenshots_1.6/hsi_7_0719_227_003_main_0/img_objectID/0000.png"
color_img_path = "/shared/mm_conv/screenshots_1.6/hsi_7_0719_227_003_main_0/img_color/0000.png"

temperature = 0.2
max_new_tokens = 512

# ============ STEP 1: Load Model ============

print("🔵 Loading model...")
model, tokenizer, image_processor, _, _ = load_pretrained_model(model_path)

# ============ STEP 2: Load object segmentation (color -> object map) ============

print("🔵 Loading object masks...")
with open(object_json_path, 'r') as f:
    object_data = json.load(f)

# Build color -> object dictionary
color_to_object = {}
for obj in object_data['objects']:
    r = int(round(obj['color'][0] * 255))
    g = int(round(obj['color'][1] * 255))
    b = int(round(obj['color'][2] * 255))
    color_to_object[(r, g, b)] = {
        "id": obj['id'],
        "name": obj['name']
    }

mask_img = Image.open(mask_img_path).convert('RGB')
mask_np = np.array(mask_img)

# ============ STEP 3: Find bounding boxes for each object ============

object_boxes = []

unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
for color in unique_colors:
    color_tuple = tuple(color)
    if color_tuple not in color_to_object:
        continue

    mask = np.all(mask_np == color_tuple, axis=-1)
    indices = np.argwhere(mask)
    if indices.size == 0:
        continue

    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)

    obj_info = color_to_object[color_tuple]

    object_boxes.append({
        "bbox": [x_min, y_min, x_max, y_max],  # pixel coordinates
        "name": obj_info['name'],
        "id": obj_info['id']
    })

print(f"🟢 Found {len(object_boxes)} objects!")

# ============ STEP 4: Randomly pick an object ============

chosen_obj = random.choice(object_boxes)
print(f"🎯 Randomly selected object: {chosen_obj['name']} (ID {chosen_obj['id']})")
bbox = chosen_obj['bbox']

# ============ STEP 5: Prepare prompt ============

color_img = Image.open(color_img_path).convert("RGB")
width, height = color_img.size

# Normalize bbox (0-1 scale)
bbox_normalized = [
    bbox[0] / width,
    bbox[1] / height,
    bbox[2] / width,
    bbox[3] / height
]

# You can also crop image to the bbox if you want a tight input:
# cropped_img = color_img.crop(bbox)

# Prepare input for model
print("🔵 Preparing input...")
image = load_image_square(color_img_path, image_processor)
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

phrase = chosen_obj['name']
utterance = f"Find the object '{phrase}' in the image."
inp = f"Given the following bbox: {bbox} marking the object {chosen_obj['name']} Geneate sentence incorporating a demonstrative  from [This, That] and the object {chosen_obj['name']}  'And I have not told you, but that table I have inherited from my mother.' "

conv = conversation_lib.default_conversation.copy()
roles = conv.roles

if model.config.mm_use_im_start_end:
    inp = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len + DEFAULT_IMAGE_END_TOKEN + '\n' + inp
else:
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

conv.append_message(roles[0], inp)
conv.append_message(roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

# ============ STEP 6: Inference ============

print("🔵 Running inference...")
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[stopping_criteria]
    )

outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
outputs = postprocess_output(outputs, color_img_path)

if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)]

print(f"📝 Model Output: {outputs}")

# ============ STEP 7: Save visualization ============

output_dir = "./viz_random_object"
os.makedirs(output_dir, exist_ok=True)

draw = ImageDraw.Draw(color_img)

# Draw model prediction box
try:
    model_bbox = np.array(ast.literal_eval(outputs))
    x1, y1, x2, y2 = model_bbox
    x1 *= width
    y1 *= height
    x2 *= width
    y2 *= height
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
except:
    print("❗ Could not parse model output")

# Draw GT box
draw.rectangle(bbox, outline="yellow", width=3)

# Save
out_path = os.path.join(output_dir, f"result.png")
color_img.save(out_path)
print(f"✅ Saved visualization to {out_path}")
