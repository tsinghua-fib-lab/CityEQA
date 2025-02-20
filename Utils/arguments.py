import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='CityEQA')

    # General Arguments
    parser.add_argument("--dataset", type=Path, default="./Data/CityEQA_EC_200.json")
    # parser.add_argument("--dataset_raw", type=Path, default="./Data/CityEQA_EC_raw.json")
    parser.add_argument("--output_directory", type=Path, default="./Results")
    parser.add_argument("--dry-run", action="store_true", default="True",
                        help="only process the first 2 questions")
    # LLM
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument("--model", type=str, default="gpt-4o")

    # experiment setting
    parser.add_argument('--collector_no_move', type=bool, default=True)

    # camera parameter
    parser.add_argument('--camera_fov', type=tuple, default=90)
    parser.add_argument('--camera_width', type=tuple, default=640)
    parser.add_argument('--camera_height', type=tuple, default=480)

    # Mapping
    # parser.add_argument('--map_origin', type=tuple, default=(-4300, 6300))
    # parser.add_argument('--map_max', type=tuple, default=(-4100, 6500))
    parser.add_argument('--map_radius', type=int, default=200)
    parser.add_argument('--map_resolution', type=int, default=1)
    parser.add_argument('--vision_range', type=int, default=90)
    parser.add_argument('--cat_threshold', type=int, default=60)
    parser.add_argument('--camera_height_range', type=float, default=2.5)

    # grounding & segment
    parser.add_argument('--landmark_list', type=list, default=['building'])
    parser.add_argument("--ground_config", type=str,
                        default="GroundSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        required=False, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="GroundSAM/groundingdino_swint_ogc.pth",
        required=False, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="GroundSAM/sam_vit_h_4b8939.pth", required=False, help="path to sam checkpoint file"
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False,
                        help="bert_base_uncased model path, default=False")

    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)


    return args
