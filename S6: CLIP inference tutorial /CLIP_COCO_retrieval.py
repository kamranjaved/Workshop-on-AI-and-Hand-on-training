import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# Path to val2014 images
image_folder = "/home/coop2025/Documents/COOP/Sara/coco_val2014v2"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()

# Load dataset
dataset = load_dataset("yerevann/coco-karpathy", split="test")

# Step 1: Encode all 5K images
image_features = []
image_ids = []

# Step 0: Build a mapping from each caption to its image ID (cocoid)
caption_to_cocoid = {}

for example in dataset:
    cocoid = example["cocoid"]
    captions = example["sentences"]  # list of captions
    for caption in captions:
        caption_to_cocoid[caption] = cocoid
        
print("Encoding all 5K images...")
for example in tqdm(dataset):
    cocoid = example["cocoid"] 
    image_path = os.path.join(image_folder, f"COCO_val2014_000000{cocoid:06d}.jpg")

    if not os.path.exists(image_path):
        continue

    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
            image_features.append(feat.cpu())
            image_ids.append(cocoid)
    except Exception as e:
        print(f"Error with image {cocoid}: {e}")

image_features = torch.cat(image_features)
image_features = image_features.to(device)
print(f"Encoded {len(image_features)} images.\n")

# Step 2: For each caption (5 per image), compute retrieval
r1 = 0
r5 = 0
r10 = 0
total = 0

print("Running caption-based image retrieval (25K queries)...")
for example in tqdm(dataset):
    cocoid = example["cocoid"] # retrieve this ground truth
    for caption in example["sentences"]:
        prompt = f"A photo of {caption}"
        text_token = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            text_feat = model.encode_text(text_token)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        sims = (text_feat @ image_features.T).squeeze(0)
        topk = sims.topk(10)
        top_image_ids = [image_ids[i] for i in topk.indices.tolist()]

        # Compute Recall@K
        if cocoid == top_image_ids[0]:
            r1 += 1
        if cocoid in top_image_ids[:5]:
            r5 += 1
        if cocoid in top_image_ids[:10]:
            r10 += 1
        total += 1


# Step 3: Report Final Results
print("\n===== Final CLIP Benchmark Results on MSCOCO 5K Test Split =====")
print(f"Total Queries (5 captions × 5K images): {total}")
print(f"Recall@1  = {r1/total:.2%}  ({r1}/{total})")
print(f"Recall@5  = {r5/total:.2%}  ({r5}/{total})")
print(f"Recall@10 = {r10/total:.2%}  ({r10}/{total})")
