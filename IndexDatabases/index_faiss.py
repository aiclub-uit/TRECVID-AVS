import faiss
import numpy as np
import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing import Union
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-features", type=str, required=True)
    parser.add_argument("--path-frame", type=str, required=True)
    parser.add_argument("--path-save", type=str, required=True)
    args = parser.parse_args()

    features_root = Path(args.path_features)
    frames_root = Path(args.path_frame) / "keyframes"

    save_index_path = Path(args.path_save)
    save_index_path.mkdir(parents=True, exist_ok=True)

    vector_dim = None
    video_paths = natsorted(features_root.iterdir())

    shot_id = 1
    index = 0

    results = {
        "index": [],
        "video_id": [],
        "shot_id": [],
        "shot_id_per_video": [],
        "frame_id": [],
    }

    for _, video_path in tqdm(enumerate(natsorted(video_paths))):
        print("Processing video %s" % video_path.stem)

        for feature_path in natsorted(video_path.iterdir()):
            feature = np.load(feature_path).astype(np.float32)
            frames_path = natsorted(
                (frames_root / video_path.stem / feature_path.stem).glob("*")
            )

            for frame_id in range(len(feature)):
                results["index"].append(index)
                results["video_id"].append(video_path.stem)
                results["shot_id"].append(shot_id)
                results["frame_id"].append(frames_path[frame_id].stem)

                # in folder extracted video, shot id is started from 0, save for later retrieval
                results["shot_id_per_video"].append(feature_path.stem)
                index += 1

            shot_id += 1

            if vector_dim is None:
                vector_dim = feature.shape[-1]
                index_faiss = faiss.IndexFlatL2(vector_dim)

            if len(feature) == 0:
                continue

            if feature.ndim == 3:
                feature = feature.reshape(-1, feature.shape[-1])

            print("Features shape: %s" % str(feature.shape))
            index_faiss.add(feature)

    faiss.write_index(index_faiss, str(save_index_path / "index.bin"))

    df = pd.DataFrame(results)
    df.to_csv(save_index_path / "mapping.csv", index=False)

    print("Total keyframe: %s" % index_faiss.ntotal)


if __name__ == "__main__":
    main()
