import json
from pathlib import Path
import torch
from torchvision.utils import save_image
from min_dalle import MinDalle

# Config
PROMPTS_FILE = "samples/input/exp_prompts.json"
OUTPUT_DIR = Path("dalle-output/explicit")
MODEL_NAME = "dalle-mini/dalle-mega"
NUM_IMAGES = 502

OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MinDalle(
    models_root="./pretrained",
    dtype=torch.float32,
    device=device,
    is_mega=True if "mega" in MODEL_NAME else False,
    is_reusable=True
)

# Load prompts
prompts = []
with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)[:NUM_IMAGES]
    
print(f"Loaded {len(prompts)} prompts.")

# Generate
for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/{len(prompts)}] Generating: {prompt}")
    
    image = model.generate_image(
        text=prompt,
        seed=-1,
        grid_size=1,
        is_seamless=False,
        temperature=1,
        top_k=256,
        supercondition_factor=32,
        is_verbose=False
    )

    # save image
    out_path = OUTPUT_DIR / f"image_{i:03d}.png"
    image.save(out_path)
    print(f"Save to {out_path}")

print("\nDONE!")