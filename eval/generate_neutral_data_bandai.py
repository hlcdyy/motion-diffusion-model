# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
from utils.fixseed import fixseed
from utils.parser_util import finetune_inpainting_style_args
from utils import dist_util
from train.training_loop import TrainInpaintingLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_serval_diffusion, creat_ddpm_ddim_diffusion
import torch 
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.tensors import collate
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.common.bvh_utils import remove_fs, fit_joints_bvh
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from visualize.vis_utils import joints2bvh
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.common.skeleton import Skeleton
from numpy.lib.format import open_memmap
import pickle
import numpy as np
from tqdm import tqdm


def main():
    args = finetune_inpainting_style_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.dataset == "stylexia_posrot":
        content_id_mapping = {
            "walking": 0,
            "running": 1,
            "jumping":2,
            "punching":3,
            "kicking":4,
            "transitions":5
        }
        style_id_mapping = {
            "angry": 0,
            "childlike":1,
            "depressed":2,
            "neutral":3,
            "old":4,
            "proud":5,
            "sexy":6,
            "strutting":7
        }

    elif args.dataset == "bandai-2_posrot":
        content_id_mapping = {
            "walk":0,
            "walk-turn-left":1,
            "walk-turn-right":2,
            "run":3,
            "wave-both-hands":4,
            "wave-left-hand":5,
            "wave-right-hand":6,
            "raise-up-both-hands":7,
            "raise-up-left-hand":8,
            "raise-up-right-hand":9,
            }

        style_id_mapping = {
            "active":0,
            "elderly":1,
            "exhausted":2,
            "feminine":3,
            "masculine":4,
            "normal":5,
            "youthful":6
        }
        id_style_mapping = ["active",
            "elderly",
            "exhausted",
            "feminine",
            "masculine",
            "normal",
            "youthful"] 
                    
    # FIXME explain
    class InpaintingTrainLoop(TrainInpaintingLoop):
        def _load_optimizer_state(self):
            pass

        def save(self):
            pass
    
    dist_util.setup_dist(args.device)

    print("creating data loader...")
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:    
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm_forstyledataset import StyleDiffusion
        from data_loaders.bandai_posrot_utils import get_inpainting_mask, BVH_JOINT_NAMES
        skeleton = paramUtil.bandai_kinematic_chain
        real_offset = paramUtil.bandai_real_offsets
        ee_names = ["Toes_R", 'Toes_L', 'Foot_L', 'Foot_R']
        anim = Skeleton(torch.Tensor(paramUtil.bandai_raw_offsets), skeleton, dist_util.dev())
    elif args.dataset == "stylexia_posrot":
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm_forstyledataset import StyleDiffusion
        from data_loaders.stylexia_posrot_utils import get_inpainting_mask, BVH_JOINT_NAMES
        skeleton = paramUtil.xia_kinematic_chain
        real_offset = paramUtil.xia_real_offsets
        ee_names = ["rtoes", 'ltoes', 'lfoot', 'rfoot']
        anim = Skeleton(torch.Tensor(paramUtil.xia_raw_offsets), skeleton, dist_util.dev())
    else:
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm import StyleDiffusion
        from data_loaders.humanml_utils import get_inpainting_mask, BVH_JOINT_NAMES
        skeleton = paramUtil.t2m_kinematic_chain
        real_offset = paramUtil.smpl_real_offsets
        ee_names = ["R_Ankle", "L_Ankle", "L_Foot", "R_Foot"]
        anim = Skeleton(torch.Tensor(paramUtil.smpl_raw_offsets), skeleton, dist_util.dev())

    max_frames = 196 if args.dataset in ['kit', 'humanml', 'bandai-1_posrot', 'bandai-2_posrot'] else 60
    max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames  

    if args.dataset == 'stylexia_posrot':
        from dataset.stylexia_split import test_list
    elif args.dataset == 'bandai-2_posrot':
        from dataset.bandai2_split import test_list
    
    content_style = {}
    for file in test_list:
        cc = file.split("_")[-3]
        ss = file.split("_")[-2]
        content_style.setdefault(cc+"_"+ss, []).append(file)

    for key, vals in content_style.items():
        assert len(vals) == 6

        
    # exist_files = os.listdir(args.save_dir)
    # exist_dict = {}
    # for exist_file in exist_files:
    #     end_index = exist_file.index("_Content")
    #     start_index = exist_file.index('dataset')
    #     exist_style_name = exist_file[start_index:end_index]
    #     exist_dict.setdefault(exist_style_name, []).append(1)
    
    # for key, value in exist_dict.items():
    #     if len(value) < 9:
    #         del exist_dict[key]
    # print(len(exist_dict))

    all_trans_motion = []
    all_trans_motion_wo_ik = []
    all_content_labels = []
    all_style_labels = []
    all_names = []
    content_id_mapping = {
            "walk":0,
            "walk-turn-left":1,
            "walk-turn-right":2,
            "run":3,
            "wave-both-hands":4,
            "wave-left-hand":5,
            "wave-right-hand":6,
            "raise-up-both-hands":7,
            "raise-up-left-hand":8,
            "raise-up-right-hand":9,
            }

    for num_file, file in enumerate(test_list):
        # if file[:-4] in exist_dict.keys():
        #     continue
        args.style_file = file

        if args.dataset == 'humanml':
            motion_dir = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs'
            joint_num = 22
        elif args.dataset == 'bandai-2_posrot':
            motion_dir = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2_with_rotation/new_joint_vecs'
            joint_num = 21
        elif args.dataset == 'stylexia_posrot':
            motion_dir = '/data/hulei/Projects/Style100_2_HumanML/style_xia_with_rotation/new_joint_vecs'
            joint_num = 20
        style_path = os.path.join(motion_dir, args.style_file)
        
        if args.save_dir is None:
            raise FileNotFoundError('save_dir was not specified.')
        elif os.path.exists(args.save_dir) and not args.overwrite:
            raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
        elif not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
        print("creating model and diffusion...")
        DiffusionClass = InpaintingGaussianDiffusion
        if args.use_ddim:
            model, inpainting_diffusion, inpainting_diffusion_ddpm = creat_ddpm_ddim_diffusion(args, ModelClass=StyleDiffusion, timestep_respacing="ddim20")
        else:
            model, inpainting_diffusion, inpainting_diffusion_ddpm = creat_ddpm_ddim_diffusion(args, ModelClass=StyleDiffusion)
        model.to(dist_util.dev())
        model.eval()

        input_style_motions, m_length = data.dataset.t2m_dataset.process_np_motion(style_path)
        input_style_motions = torch.Tensor(input_style_motions.T).unsqueeze(1).unsqueeze(0)
        input_style_motions = input_style_motions.to(dist_util.dev()) 

        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1

        if args.dataset == 'bandai-2_posrot':
            contents = args.style_file.split("_")[-3].split("-")
            content_id = content_id_mapping[args.style_file.split("_")[-3]]
            s_c = args.style_file.split("_")[-3]
            s_s = args.style_file.split("_")[-2]
            dict_index = content_style[s_c+"_"+s_s].index(args.style_file)
            s_index = style_id_mapping[s_s]
            next_index = (s_index + 1) % 7
            style_label = args.style_file.split("_")[-2]
            contents[0] += "s"
            contents = " ".join(contents)
            caption = 'a person ' +  contents + " normal"
        elif args.dataset == 'stylexia_posrot':
            contents = args.style_file.split("_")[-1][:-4]
            style_label = args.style_file.split("_")[0][3:]
            caption = 'a person is ' + contents + " neutral"

        if style_label in ["normal", 'neutral']:
            continue
        texts = [caption] * 1
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)    
        model_kwargs['y']['inpainted_motion'] = input_style_motions  
        model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_style_motions.shape)).float().to(dist_util.dev())
        model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * 2.5

        if args.use_ddim:
            stop_timesteps = int(900/1000 * 20)
            sample_fn = inpainting_diffusion.ddim_sample_loop
        else:
            stop_timesteps = 900
            sample_fn = inpainting_diffusion.p_sample_loop
        with torch.no_grad():
            net = model.motion_enc.mdm_model
            sample = sample_fn(
                net,
                (1, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=input_style_motions,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                stop_timesteps=stop_timesteps,
                dump_all_xstart=True,
            )
            sample = sample[-1]
        
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample_feature = sample.clone()
        sample = recover_from_ric(sample, joint_num)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        sample = sample.cpu().numpy()

        name = "Neutral_" + args.style_file[:-4]
        all_names.append(name)
        all_style_labels.append(5)
        all_content_labels.append(content_id)
        fs_motion = sample[0].transpose(2, 0, 1)[:m_length].copy() # T J 3
        save_bvh = name + '_woik.bvh'
        bvh_save_path = os.path.join(args.save_dir, save_bvh)
        # fit_joints_bvh(bvh_save_path, sample_feature[0, 0, :m_length], joint_num, anim, real_offset, fs_motion, BVH_JOINT_NAMES)
        all_trans_motion_wo_ik.append(fs_motion)
        fs_motion, _, _, _ = remove_fs("", fs_motion, fs_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False)
        
        save_bvh = name + '.bvh'
        bvh_save_path = os.path.join(args.save_dir, save_bvh)
        fit_joints_bvh(bvh_save_path, sample_feature[0, 0, :m_length], joint_num, anim, real_offset, fs_motion, BVH_JOINT_NAMES)
        all_trans_motion.append(fs_motion)

    
    max_frame_eval = 350
    fp = open_memmap(
        '{}/{}_data.npy'.format(args.save_dir, "output"),
        dtype='float32',
        mode='w+',
        shape=(len(all_trans_motion), 3, max_frame_eval, joint_num, 1))
    
    fp_woik = open_memmap(
        '{}/{}_woik_data.npy'.format(args.save_dir, "output"),
        dtype='float32',
        mode='w+',
        shape=(len(all_trans_motion_wo_ik), 3, max_frame_eval, joint_num, 1))
    
    for i in range(len(all_trans_motion)):
        data = all_trans_motion[i].transpose(2, 0, 1)
        fp[i, :, 0:data.shape[1], :, 0] = data
        
        data_woik = all_trans_motion_wo_ik[i].transpose(2, 0, 1)
        fp_woik[i, :, 0:data_woik.shape[1], :, 0] = data_woik
    
    
    with open('{}/{}_label.pkl'.format(args.save_dir, "content"), 'wb') as f:
        pickle.dump((all_names, all_content_labels), f)
    
    with open('{}/{}_label.pkl'.format(args.save_dir, "style"), 'wb') as f:
        pickle.dump((all_names, all_style_labels), f)
      

       
        

if __name__ == "__main__":
    main()
