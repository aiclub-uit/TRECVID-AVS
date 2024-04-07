from pathlib import Path
from tqdm import tqdm
from PIL import Image
from natsort import natsorted

import argparse
import numpy as np
import torch
import clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-frame", type=str, default=None)
    parser.add_argument("--path-save", type=str, default=None)
    args = parser.parse_args()

    assert args.path_frame is not None, "Please provide path to frame"
    assert args.path_save is not None, "Please provide save path"

    kf_root = Path(args.path_frame) / "keyframes"
    save_root = Path(args.path_save)
    save_root.mkdir(exist_ok=True, parents=True)

    # CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    for video_path in tqdm(natsorted(list(kf_root.iterdir()))):
        video_id = video_path.stem
        save_path = save_root / video_id
        save_path.mkdir(exist_ok=True, parents=True)

        for shot_path in natsorted(list(video_path.iterdir())):
            shot_id = shot_path.stem
            images = []

            for frame_path in natsorted(list(shot_path.iterdir())):
                images.append(preprocess(Image.open(frame_path)))

            with torch.no_grad():
                if len(images):
                    processed_images = torch.stack(images).to(device)
                    image_features = model.encode_image(processed_images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features.cpu().detach().numpy()
                else:
                    image_features = np.array([])

            np.save(save_path / f"{shot_id}.npy", image_features)
