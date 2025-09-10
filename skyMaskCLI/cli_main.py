import os, torch

caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{m}.{n}" for m, n in sorted(caps))

import argparse
from skyMaskMain.mask_main import MaskProcessor

def main():
    parser = argparse.ArgumentParser(prog="skymask", description="skyMask image processing tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mask = sub.add_parser("process-masks", help="Generate binary masks for images from user defined prompt")
    p_mask.add_argument("--data-dir", "-d", required=True,
                        help="Path to images directory")
    p_mask.add_argument("--prompt", "-p", default="sky",
                        help="Detection prompt (default: 'sky')")
    p_mask.add_argument("--box-threshold", "-bt", default=".5",
                        help="theshold for bounding box detections")
    p_mask.add_argument("--text-threshold", "-tt", default=".25",
                        help="threshold for words inside the label")
    p_mask.add_argument("--inspect", "-i", default=False,
                        help="view mask of first image and every 10th image afterawards")

    args = parser.parse_args()
    if args.command == "process-masks":
        mp = MaskProcessor(args.data_dir, prompt=args.prompt, inspect=args.inspect)
        mp.run_mask_processing(float(args.box_threshold), float(args.text_threshold))

with torch.inference_mode():
    if __name__ == "__main__":
        main()