import cv2

import ffmpeg
import subprocess
import numpy as np
import pandas as pd
import argparse

from PIL import Image
from pathlib import Path
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from natsort import natsorted


# Initiate ORB detector
orb = cv2.ORB_create()

# Since we use ORB, we should use hamming distance
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# hack for steam.run_async(quiet=True) bug
def _run_async_quiet(stream_spec):
    args = ffmpeg._run.compile(stream_spec, "ffmpeg", overwrite_output=False)
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def extract_all_frames(video_filename, output_width, output_height):
    video_stream, err = (
        ffmpeg.input(video_filename)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(output_width, output_height),
        )
        .run(capture_stdout=True, capture_stderr=True)
    )

    return np.frombuffer(video_stream, np.uint8).reshape(
        [-1, output_height, output_width, 3]
    )


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def get_video_stream(video_path, output_width, output_height):
    return _run_async_quiet(
        ffmpeg.input(video_path).output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(output_width, output_height),
        )
    )


def extract_keyframe(
    video_path,
    thres=0.9,
    output_width=400,
    output_height=225,
    skip_frame=10,
):
    # last_endframe = msb.iloc[-1, 2]
    video_id = video_path.with_suffix("").name

    # Open video stream buffer
    video_stream = get_video_stream(video_path, output_width, output_height)

    processed_frames = 0
    total_kf = 0

    frame_idx = 0

    if video_path.stem in DONE_LIST:
        print(f"Skip {video_path}")
        return frame_idx, processed_frames, total_kf

    cnt = 0

    print(f"Processing {video_path}")

    msb = msb_dir / f"{video_id}.tsv"
    msb = pd.read_csv(msb, sep="\t")
    mapping_idx_shot = {}

    for row in msb.itertuples():
        for t in range(int(row[1]), int(row[3]) + 1):
            mapping_idx_shot[t] = row[0]

    for shot_id in range(len(msb)):
        keyframe = None
        keyframe_feature = None
        last_frame_feature = None

        while True:
            in_bytes = video_stream.stdout.read(output_width * output_height * 3)

            # End of video
            if not in_bytes:
                break

            cnt += 1

            if mapping_idx_shot[frame_idx] < shot_id:
                frame_idx += 1
                continue

            if frame_idx % skip_frame != 0:
                frame_idx += 1

                if frame_idx not in mapping_idx_shot:
                    break

                if mapping_idx_shot[frame_idx] > shot_id:
                    break

                continue

            processed_frames += 1

            # Get keyframe image name and create folder if not exists
            img_name = f"{frame_idx:06d}.jpg"
            save_img_path = (
                processed_kf
                / str(video_id)
                / str(mapping_idx_shot[frame_idx])
                / img_name
            )
            save_img_path.parent.mkdir(parents=True, exist_ok=True)

            frame = np.frombuffer(in_bytes, np.uint8).reshape(
                [output_height, output_width, 3]
            )

            # Convert frame to PIL Image
            img = Image.fromarray(frame)

            # ORB Feature
            cv_img = np.array(img.convert("L"))
            kp = orb.detect(cv_img)
            kp, des = orb.compute(cv_img, kp)
            frame_feature = (kp, des)

            if des is None:
                # Skip frame without any keypoint
                frame_idx += 1

                if frame_idx not in mapping_idx_shot:
                    break

                if mapping_idx_shot[frame_idx] > shot_id:
                    break

                continue

            # Select the first frame as the keyframe
            if keyframe is None:
                keyframe = frame.copy()
                keyframe_feature = (kp, des)
                img.save(str(save_img_path))
                total_kf += 1
            else:
                # Compare similarity of current frame vs last kf and last frame
                matches = matcher.knnMatch(last_frame_feature[1], frame_feature[1], k=2)
                good = []

                for pair in matches:
                    if len(pair) != 2:
                        continue

                    m, n = pair
                    if m.distance < 0.7 * n.distance:
                        good.append([m])

                last_frame_similar = 1 - len(good) / max(
                    len(frame_feature[0]), len(last_frame_feature[0])
                )

                matches = matcher.knnMatch(keyframe_feature[1], frame_feature[1], k=2)
                good = []

                for pair in matches:
                    if len(pair) != 2:
                        continue

                    m, n = pair
                    if m.distance < 0.7 * n.distance:
                        good.append([m])

                kf_similar = 1 - len(good) / max(
                    len(frame_feature[0]), len(keyframe_feature[0])
                )

                if kf_similar > thres or last_frame_similar > thres:
                    keyframe = frame.copy()
                    keyframe_feature = frame_feature
                    img.save(str(save_img_path))
                    total_kf += 1

            # Update last frame
            last_frame_feature = (kp, des)
            frame_idx += 1

            if frame_idx not in mapping_idx_shot:
                break

            if mapping_idx_shot[frame_idx] > shot_id:
                break

    print("Total frames: ", cnt)
    return frame_idx, processed_frames, total_kf


def find_video_file(basePath, video_id, supported_exts):
    for ext in supported_exts:
        video_path = basePath / video_id / (video_id + ext)

        if video_path.exists():
            return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--msb-dir", required=True, type=str)
    parser.add_argument("--path-save", type=str, default="extracted_shot")
    parser.add_argument("--skip-frame", type=int, default=10)
    parser.add_argument("--resize", action="store_true")  # false
    args = parser.parse_args()

    # ----------------
    video_dir = Path(args.video_dir)
    msb_dir = Path(args.msb_dir)

    processed_root = Path(args.path_save)
    processed_kf = processed_root / "keyframes"
    processed_log = processed_root / "done.txt"
    processed_results = processed_root / "extracted_kf.csv"

    processed_kf.mkdir(exist_ok=True, parents=True)
    processed_root.mkdir(exist_ok=True, parents=True)

    DONE_LIST = open(processed_log, "w+").readlines()
    DONE_LIST = [x.strip("\n") for x in DONE_LIST]

    supported_exts = [".mp4", ".avi", ".m4v", ".mov", ".mpe", ".vtt"]
    total_frames = 0
    processed_frames = 0
    total_kf = 0
    skip_frame = args.skip_frame
    # ----------------

    process_videos = []
    for video_folder_path in natsorted(video_dir.iterdir()):
        video_id = video_folder_path.name

        try:
            int(video_id)
        except:
            print("Skip folder %s" % video_id)
            continue

        if not (int(video_id) >= args.start and int(video_id) < args.end):
            continue

        process_videos.append(video_id)

    for video_id in tqdm(process_videos):
        video_id = video_id.split(".")[0]
        video_path = find_video_file(video_dir, video_id, supported_exts)

        if not video_path:
            print("Video %s not found. Skip it" % video_id)
            continue

        if args.resize:
            n_frames, n_procf, n_kf = extract_keyframe(
                video_path, output_width=1280, output_height=720
            )
        else:
            n_frames, n_procf, n_kf = extract_keyframe(video_path)

        # Save processed video to log
        with open(processed_log, "a+") as f:
            f.write(str(video_path.stem) + "\n")

        total_frames += n_frames
        processed_frames += n_procf
        total_kf += n_kf

    print("Total frame: %s" % total_frames)
    print("Total processed frame: %s" % processed_frames)
    print("Total keyframe extracted: %s" % total_kf)
