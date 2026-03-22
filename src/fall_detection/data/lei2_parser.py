import re
from pathlib import Path
import pandas as pd

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
TEXT_EXTS = {".txt", ".csv", ".json"}

def is_video_file(path: Path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS

def is_text_file(path: Path):
    return path.is_file() and path.suffix.lower() in TEXT_EXTS

def find_scene_leaf_dirs(dataset_root: Path):
    leaf_dirs = []
    for p in dataset_root.rglob("*"):
        if not p.is_dir():
            continue
        child_names = {x.name.lower() for x in p.iterdir() if x.is_dir()}
        if "videos" in child_names and "annotation_files" in child_names:
            leaf_dirs.append(p)
    return sorted(leaf_dirs)

def resolve_videos_dir(scene_dir: Path):
    for p in scene_dir.iterdir():
        if p.is_dir() and p.name.lower() == "videos":
            return p
    return None

def resolve_annotations_dir(scene_dir: Path):
    for p in scene_dir.iterdir():
        if p.is_dir() and p.name.lower() == "annotation_files":
            return p
    return None

def extract_all_integers(text):
    return [int(x) for x in re.findall(r"\d+", text)]

def parse_le2i_annotation_file(annotation_path: Path):
    text = annotation_path.read_text(encoding="utf-8", errors="ignore").strip()
    nums = extract_all_integers(text)
    if len(nums) == 0:
        return {"fall_start": None, "fall_end": None, "lying_start": None}
    
    if len(nums) == 1:
        fall_start = fall_end = nums[0]
    else:
        fall_start, fall_end = nums[0], nums[1]
        if fall_end < fall_start:
            fall_start, fall_end = fall_end, fall_start
            
    return {
        "fall_start": int(fall_start),
        "fall_end": int(fall_end),
        "lying_start": int(fall_end + 1)
    }

def build_annotation_map(annotation_dir: Path):
    ann_map = {}
    for ann_file in sorted(annotation_dir.iterdir()):
        if not is_text_file(ann_file):
            continue
        parsed = parse_le2i_annotation_file(ann_file)
        ann_map[ann_file.stem.lower()] = {
            "annotation_file": str(ann_file),
            **parsed
        }
    return ann_map

def best_match_annotation(video_stem: str, ann_map: dict):
    key = video_stem.lower()
    if key in ann_map:
        return ann_map[key]
    
    norm_key = re.sub(r"[^a-z0-9]+", "", key)
    
    for k, v in ann_map.items():
        if key in k or k in key:
            return v
    for k, v in ann_map.items():
        norm_k = re.sub(r"[^a-z0-9]+", "", k)
        if norm_k == norm_key:
            return v
    return None

def build_le2i_annotations(dataset_root: Path):
    all_records = []
    scene_dirs = find_scene_leaf_dirs(dataset_root)
    
    for scene_dir in scene_dirs:
        videos_dir = resolve_videos_dir(scene_dir)
        ann_dir = resolve_annotations_dir(scene_dir)
        
        if videos_dir is None or ann_dir is None:
            continue
            
        ann_map = build_annotation_map(ann_dir)
        scene_name = scene_dir.name
        
        for video_path in sorted(videos_dir.iterdir()):
            if not is_video_file(video_path):
                continue
                
            matched_ann = best_match_annotation(video_path.stem, ann_map)
            
            if matched_ann is None:
                record = {
                    "scene": scene_name,
                    "video_path": str(video_path),
                    "annotation_file": None,
                    "fall_start": None,
                    "fall_end": None,
                    "lying_start": None
                }
            else:
                record = {
                    "scene": scene_name,
                    "video_path": str(video_path),
                    "annotation_file": matched_ann["annotation_file"],
                    "fall_start": matched_ann["fall_start"],
                    "fall_end": matched_ann["fall_end"],
                    "lying_start": matched_ann["lying_start"]
                }
            all_records.append(record)
            
    return pd.DataFrame(all_records)
