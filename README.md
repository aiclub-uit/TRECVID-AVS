## Setup Enviroment

```bash
# clone the repo
git clone https://github.com/aiclub-uit/TRECVID-AVS
cd TRECVID-AVS

# setup docker images
docker build . -t trecvid

# run docker container
docker run -it -v <mount_data_to_container> \
        --gpus <gpu_ids> trecvid bash
```

## Usage

Now you are in Docker containers.

### Extracing frames

```bash
cd /TRECVID/ExtractFrames
python extract.py \
    --video-dir <path_video_folder> \
    --msb-dir <path_master_shot_folder> \
    --skip-frame <frequency_frame_extraction> \
    --start <id_video_end> \
    --end <id_video_end> \
    --path-save <path_to_save_extracted_frames> \
```

### Extracing features

```bash
cd /TRECVID/ExtractFeatures
python clip_model.py \
    --path-frame <path_saved_extracted_frames> \
    --path-save <path_to_save_features>
```

### Indexing to Faiss

```bash
cd /TRECVID/IndexDatabases
python index_faiss.py \
    --path-features <path_saved_features> \
    --path-frame <path_saved_extracted_frames> \
    --path-save <path_to_save_faiss_data>
```

### Retrieve

```bash
python retrieve.py \
    --path-faiss ../IndexDatabases/databases/clip/ 
    --query <query_to_retrieve>
    --path-frame <path_saved_extracted_frames>
    --save-path <path_to_save_ranking_results>
```