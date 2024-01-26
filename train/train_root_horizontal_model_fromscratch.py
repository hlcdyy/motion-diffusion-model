# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from utils.fixseed import fixseed
from utils.parser_util import train_inpainting_from_scratch_args
from utils import dist_util
from train.training_loop import TrainInpaintingLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_stylediffuse_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from model.mdm_forstyledataset import MDM
import torch 
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion


def main():
    args = train_inpainting_from_scratch_args()
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
    
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        split = 'train'
    else:
        split = 'train'
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split=split)
    
    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion
    model, diffusion = creat_stylediffuse_and_diffusion(args, ModelClass=MDM, DiffusionClass=DiffusionClass)
    
    # for i, ((motion, cond), (normal_motion, normal_cond)) in enumerate(data):
    #     print(motion.shape, normal_motion.shape)
        
    # mdm_model.to(dist_util.dev())
    # diffusion.to(dist_util.dev())
    model.to(dist_util.dev())

    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        from data_loaders.bandai_posrot_utils import get_inpainting_mask
    elif args.dataset == 'stylexia_posrot':
        from data_loaders.stylexia_posrot_utils import get_inpainting_mask
    else:
        from data_loaders.humanml_utils import get_inpainting_mask

    class InpaintingDataLoader(object):
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            for motion, cond in super().__getattribute__('data').__iter__():
                if args.inpainting_mask != "":
                    cond['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).to(motion.device)
                    # do not need inpainted motion when training
                    # cond['y']['inpainted_motion'] = motion.float()
                    yield motion, cond
        
        def __getattribute__(self, name):
            return super().__getattribute__('data').__getattribute__(name)
        
        def __len__(self):
            return len(super().__getattribute__('data'))
        
    data = InpaintingDataLoader(data)

    # FIXME explain
    class InpaintingTrainLoop(TrainInpaintingLoop):
        def _load_optimizer_state(self):
            pass

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    InpaintingTrainLoop(args, train_platform, model, data, diffusion=diffusion).run_loop()
    train_platform.close()
   

if __name__ == "__main__":
    main()
