from pathlib import Path

import shutil
import torch
import pandas as pd
import argparse
import clip
import faiss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-faiss", type=str, required=True)
    parser.add_argument("--path-frame", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-retrieve", type=int, default=50)
    args = parser.parse_args()

    print("Faiss path: ", args.path_faiss)
    print("Query: ", args.query)

    faiss_path = Path(args.path_faiss) / "index.bin"
    mapping_path = Path(args.path_faiss) / "mapping.csv"
    frame_root = Path(args.path_frame) / "keyframes"
    save_root = Path(args.save_path)
    save_root.mkdir(exist_ok=True, parents=True)

    mapping_df = pd.read_csv(mapping_path)
    index_faiss = faiss.read_index(str(faiss_path))
    print(index_faiss.ntotal)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    tokenized_query = clip.tokenize([args.query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokenized_query)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        D, I = index_faiss.search(text_features.cpu().numpy(), args.num_retrieve)
        I = I[0]

        filtered_results = mapping_df.iloc[I]

        for ranking, (_, row) in enumerate(filtered_results.iterrows()):
            ranking = str(ranking).zfill(6)
            video_id = str(row["video_id"]).zfill(5)
            frame_id = str(row["frame_id"]).zfill(6)
            shot_id = str(row["shot_id"])

            # this is NOT shot_id
            shot_id_per_video = str(row["shot_id_per_video"])

            # copy to results folder for visualization
            source_path = (
                frame_root / video_id / shot_id_per_video / (frame_id + ".jpg")
            )

            destination_path = (
                save_root / f"{ranking}_{video_id}_{shot_id}_{frame_id}.jpg"
            )

            shutil.copy(source_path, destination_path)
