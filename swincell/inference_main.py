import numpy as np
from monai import data, transforms
from swincell.utils.device_utils import get_device
from swincell.utils.utils import load_default_config
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from functools import partial
from monai.inferers import sliding_window_inference
import tifffile
from swincell.cellpose_dynamics import compute_masks
from monai.networks.nets import SwinUNETR
import torch
import os
import argparse
from glob import glob
from swincell.utils.utils import get_random_cmap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend good for savefig
import matplotlib.pyplot as plt
from tqdm import tqdm
parser = argparse.ArgumentParser(description="SwinCell Training")
parser.add_argument("--datadir", default=None, help="Dataset path")
parser.add_argument("--a_min", default=0, type=float, help="cliped min input value")
parser.add_argument("--a_max", default=255,type=float, help="clipped max input value")
parser.add_argument("--b_min", default=0, type=float, help="min target (output) value")
parser.add_argument("--b_max", default=1,type=float, help="Umax target value")
parser.add_argument("--dataset", default="nanolive", type=str, help="dataset name")
parser.add_argument("--save_logits", default=False, type=bool, help="Save pred logits")
parser.add_argument("--downsample_factor", default=2, type=float, help="Iptimization learning rate")
parser.add_argument("--model_path", default="", type=str, help="Path to model weight")
parser.add_argument("--outdir", default="./", type=str, help="Output directory path")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels, #cell probability channel + #flow channels")
parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="Sliding window inference overlap")
parser.add_argument("--save_flows", action="store_true", help="Save predicted flows")
parser.add_argument("--save_preview", action="store_true", help="Save preview of prediction")
args = parser.parse_args()
infer_ROI = (256,256,32)
device = get_device()
model = SwinUNETR(in_channels=args.in_channels, out_channels=args.out_channels, use_checkpoint=True,feature_size=args.feature_size)
ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
state_dict = ckpt["state_dict"]
model.load_state_dict(state_dict)
model.to(device)
model.eval()
if not os.path.exists(args.outdir):
    os.makedirs(parser.outdir)

image_paths = sorted(glob(os.path.join(args.datadir+"/images/", "*"))) 
label_paths = sorted(glob(os.path.join(args.datadir+"/labels/", "*"))) 
test_datalist = [
    {"image": img, "label": lbl}
    for img, lbl in zip(image_paths, label_paths)
]
if args.dataset =='colon':
    img_shape= (1300,1030,129) #original shape
    img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
    img_reshape = tuple(int(e) for e in img_reshape)

elif args.dataset =='allen':
    img_shape=(900,600,64)
    img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
    img_reshape = tuple(int(e) for e in img_reshape)

elif args.dataset =='nanolive':
    img_shape=(512,512,96)
    img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
    img_reshape = tuple(int(e) for e in img_reshape)

else:
    raise Warning("dataset not defined")
    img_reshape = None
    
test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Resized(keys=["image", "label"],spatial_size=img_reshape),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),

            transforms.ToTensord(keys=["image", "label"]),
        ]
    )


test_ds = data.Dataset(data=test_datalist, transform=test_transform)
test_loader = data.DataLoader(
        test_ds, batch_size=1, sampler=None, drop_last=False
    )

model_inferer = partial(
    sliding_window_inference,
    roi_size=infer_ROI,
    sw_batch_size=2,
    predictor=model,
    overlap=0.5,
    mode='gaussian'
)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
with torch.no_grad():
        for idx, batch_data in tqdm(enumerate(test_loader)):
            out_filename = test_datalist[idx]['image'].split('.')[0].split('/')[-1] +'_pred.tiff' 
            data_test = batch_data["image"]
            data_test = data_test.to(device)
    
            logits = model_inferer(data_test)         
            logits_out =  np.squeeze(logits.detach().cpu().numpy())
            logits_out[0] = post_pred(post_sigmoid(logits_out[0]))

            logits_out_transposed = np.transpose(logits_out,(0,3,2,1))
            flows = logits_out[1:4,:,:,:]
            if args.save_flows:
                tifffile.imwrite(
                    os.path.join(args.output_dir, f'logits_transposed_{out_filename}'),
                    logits_out_transposed
                )

            tifffile.imwrite(
                os.path.join(args.output_dir, f'masks_{out_filename}'),
                masks_recon
            )
            
            masks_recon,p = compute_masks(logits_out_transposed[[3,2,1],:,:,:],logits_out_transposed[0,:,:,:],cellprob_threshold=0.4,flow_threshold=0.4, do_3D=True,min_size=2500//args.downsample_factor//args.downsample_factor, use_gpu=True if device.type =="cuda" else False)
            
            print(masks_recon.shape)

if args.save_preview:
    n_row = 2
    fig, axes = plt.subplots(2, 2,sharex=False, sharey=False, figsize=(12,10))
    img_shape = logits.shape
    slice2view = int(img_shape[-1]//2)
    img=np.squeeze(data_test.detach().cpu().numpy())
    print(img.shape,logits_out.shape)
    flow= logits_out[1:4]
    flow_slice = flow[:,:,:,slice2view].transpose(1, 2, 0)
    print(flow.shape)


    axes[0,0].imshow(img[:,:,slice2view])
    axes[0,0].set_title('Raw')

    axes[0,1].imshow(logits_out[0,:,:,slice2view])
    axes[0,1].set_title('Cell prob')

    axes[1,0].imshow(flow_slice)
    axes[1,0].set_title('Predicted flows')                                                                                       


    axes[1,1].imshow(masks_recon[slice2view].T,cmap=get_random_cmap(30))
    axes[1,1].set_title('Predicted Masks')

    for i in range(2):
        for j in range(2):
            axes[i,j].axis('off')

    plt.savefig(os.path.join(args.outdir, "output_example.png"))