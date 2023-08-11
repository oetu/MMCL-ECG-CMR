import argparse
import torch

import os

def get_args_parser():
    parser = argparse.ArgumentParser('Get the weights of the signal and image encoder', add_help=False)
    # Basic parameters
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='checkpoint path (default: None)')
    
    return parser

def main(args):
    mm_checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    print("Load pre-trained checkpoint from: %s" % args.checkpoint_path)
    
    signal_encoder = {(".").join(key.split(".")[1:]):value for key, value in mm_checkpoint["state_dict"].items() if "encoder_ecg" in key}
    image_encoder = {key:value for key, value in mm_checkpoint["state_dict"].items() if "encoder_imaging" in key}

    # remove head from the VIT
    k = ["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"]
    for key in k:
        try:
            del signal_encoder[key]
        except:
            pass

    torch.save({"model":signal_encoder}, os.path.join(os.path.split(args.checkpoint_path)[0], "signal_encoder.pth"))
    torch.save({"state_dict":image_encoder}, os.path.join(os.path.split(args.checkpoint_path)[0], "image_encoder.pth"))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    main(args)