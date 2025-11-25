import os
import argparse
import json
import numpy as np
import torch
import tifffile
from functools import partial
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations
from swincell.utils.utils import load_model
from swincell.utils.data_utils import folder_loader
from swincell.cellpose_dynamics import compute_masks


def get_default_args():
    parser = argparse.ArgumentParser(description="SwinCell Inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', type=str, help='Folder containing data.')
    parser.add_argument("--output_dir", default='./results', help="Folder to save results")
    parser.add_argument("--model_dir", default='./results/model.pt', help="Path to saved model")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of batch size")
    parser.add_argument("--gpu", default=1, type=int, help="GPU id to use")
    parser.add_argument("--in_channels", default=1, type=int, help="Number of input channels")
    parser.add_argument("--out_channels", default=4, type=int, help="Number of output channels")
    parser.add_argument("--a_min", default=0, type=float, help="Clipped min input value")
    parser.add_argument("--a_max", default=255, type=float, help="Clipped max input value")
    parser.add_argument("--b_min", default=0.0, type=float, help="Min target (output) value")
    parser.add_argument("--b_max", default=1.0, type=float, help="Max target value")
    parser.add_argument("--roi_x", default=128, type=int, help="ROI size in x direction")
    parser.add_argument("--roi_y", default=128, type=int, help="ROI size in y direction")
    parser.add_argument("--roi_z", default=32, type=int, help="ROI size in z direction")
    parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="Sliding window inference overlap")
    parser.add_argument("--downsample_factor", default=1, type=int, help="Downsampling rate of input data")
    parser.add_argument("--model", default="swin", type=str, help="Model architecture")
    parser.add_argument("--dataset", default="colon", type=str, help="Dataset name")
    parser.add_argument("--save_flows", action="store_true", help="Save predicted flows")
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    return parser

def load_json_config(json_file):
    if json_file and os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    return {}

def parse_arguments():
    parser = get_default_args()
    args, unknown = parser.parse_known_args()

    # Load JSON config if provided
    json_config = load_json_config(args.config)

    # Update defaults with JSON config
    for key, value in json_config.items():
        if hasattr(args, key):
            parser.set_defaults(**{key: value})

    # Parse arguments again with updated defaults
    args = parser.parse_args()

    return args

def main_infer():
    args = parse_arguments()
    args.test_mode = True
    args.checkpoint = False
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.data_folder:
        infer_loader, test_image_paths = folder_loader(args)
    else:
        infer_loader = None

    model = load_model(args).to(device)
    model_dict = torch.load(args.model_dir)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=2,
        predictor=model,
        overlap=args.infer_overlap,
        mode='gaussian'
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        for idx, batch_data in enumerate(infer_loader):
            data = batch_data["image"].to(device)
            base_filename = os.path.basename(test_image_paths[idx]).split('.')[0]
            output_filename = f"{base_filename}_prediction.tiff"

            logits = model_inferer(data)
            logits_out = np.squeeze(logits.detach().cpu().numpy())
            # print("logits_out shape:", logits_out.shape)

            # Process the cell probability channel
            logits_out[0] = post_pred(post_sigmoid(logits_out[0]))

            # Transpose for further processing (adjust axes as needed)
            logits_out_transposed = np.transpose(logits_out, (0, 3, 2, 1))
            # print("logits_out_transposed shape:", logits_out_transposed.shape)

            # Compute cell masks
            masks_recon, _ = compute_masks(
                logits_out_transposed[1:4, :, :, :],
                logits_out_transposed[0, :, :, :],
                cellprob_threshold=0.4,
                flow_threshold=0.4,
                do_3D=True,
                min_size=args.min_size,
                use_gpu=True
            )

            # Optionally save predicted flows if enabled
            if args.save_flows:
                tifffile.imwrite(
                    os.path.join(args.output_dir, f'logits_transposed_{output_filename}'),
                    logits_out_transposed
                )

            # Save the reconstructed masks
            tifffile.imwrite(
                os.path.join(args.output_dir, f'masks_{output_filename}'),
                masks_recon
            )

if __name__ == "__main__":
    main_infer()