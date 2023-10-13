# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
from utils.fixseed import fixseed
from utils.parser_util import train_style_diffusion_module_args
from utils import dist_util
from train.training_loop import TrainLoopStyleDiffusion
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_stylediffuse_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from data_loaders.humanml_utils import get_inpainting_mask
import torch 
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion


def main():
    args = train_style_diffusion_module_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args. save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.style_dataset, batch_size=args.batch_size, num_frames=args.num_frames, pairs=True)
    t2m_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion
    model, diffusion = creat_stylediffuse_and_diffusion(args, DiffusionClass=DiffusionClass)
    
    
    # for i, ((motion, cond), (normal_motion, normal_cond)) in enumerate(data):
    #     print(motion.shape, normal_motion.shape)
        
    # mdm_model.to(dist_util.dev())
    # diffusion.to(dist_util.dev())
    model.to(dist_util.dev())

    class InpaintingDataLoader(object):
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            for motion, cond in super().__getattribute__('data').__iter__():
                if args.inpainting_mask != "":
                    cond['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).to(motion.device)
                    cond['y']['inpainted_motion'] = motion.float()
                else:
                    cond['y']['inpainting_mask'] = torch.zeros_like(motion).to(motion.device).float()                    
                yield motion, cond
        
        def __getattribute__(self, name):
            return super().__getattribute__('data').__getattribute__(name)
        
        def __len__(self):
            return len(super().__getattribute__('data'))
        
    t2m_data = InpaintingDataLoader(t2m_data)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_enc()) / 1000000.0))
    print("Training...")
    TrainLoopStyleDiffusion(args, train_platform, model, data, t2m_data, diffusion=diffusion).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
