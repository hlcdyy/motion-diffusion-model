# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import json
from utils.fixseed import fixseed
from utils.parser_util import finetune_inpainting_style_args
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils import dist_util
from train.training_loop import TrainInpaintingLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_stylediffuse_and_diffusion, creat_serval_diffusion, creat_ddpm_ddim_diffusion, creat_motion_trajectory, load_model_wo_clip
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch 
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.tensors import collate
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil
skeleton = paramUtil.t2m_kinematic_chain
from data_loaders.humanml.common.bvh_utils import remove_fs



def main():
    args = finetune_inpainting_style_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)

    if args.dataset == 'humanml':
        # motion_dir = '/home/synthesis/xuyiwen/TrackDnD/datasets/3DPW/estimated_v2/outdoors_fencing_01'
        # path = os.path.join(motion_dir, 'hybrik.pt')
        # motion_dir = './videos/res_xixiyu/'
        # path = os.path.join(motion_dir, 'res.pk')
        # motion_dir = './videos/H36M_S6/Hybrik_Walking_1.60457274/'
        # path = os.path.join(motion_dir, 'res.pk')
        motion_dir = './videos/100_walks/'
        path = os.path.join(motion_dir, 'res.pk')
        
        # fps = 30 # xixiyu
        # fps = 50  # /H36M_S6/Hybrik_Walking_1.60457274/
        fps = 24 # 100 walks
        # start_second = 52.4  # xixiyu 潜行
        # end_second = 55.5
        # start_second = 15  #  S6 Walking_1.60457274
        # end_second = 22
        # start_second = 2.8  # xixiyu 牛仔
        # end_second = 5
        # start_second = 56 # xixiyu tired
        # end_second = 61
        # start_second = 39.4 # xixiyu 逆风
        # end_second = 44
        start_second = 33 
        end_second = 36
        
        from utils.process_smpl_from_hybrik import amass_to_pose, pos2hmlrep
        pose_seq_np_n, joints = amass_to_pose(path, fps=fps, with_trans=False)
        path = pos2hmlrep(joints)
        # args.style_file = 'outdoors_fencing_01.npy'
        # args.style_file = 'stealthily.npy'
        # args.style_file = 'walk_dog.npy'
        # args.style_file = 'cowboy.npy'
        # args.style_file = 'tired.npy'
        # args.style_file = 'against_the_wind.npy'
        args.style_file = 'zombie.npy'
        end_frame = int(20*end_second) if int(20*end_second) < path.shape[0] else path.shape[0]
        path = path[int(20*start_second):end_frame] # after process fixed 20 fps


    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    args.save_dir = os.path.join(args.save_dir, args.style_file[:-4])

    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

   
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:    
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm_forstyledataset import StyleDiffusion
        from data_loaders.bandai_posrot_utils import get_inpainting_mask
    elif args.dataset == "stylexia_posrot":
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm_forstyledataset import StyleDiffusion
        from data_loaders.stylexia_posrot_utils import get_inpainting_mask
    else:
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm import StyleDiffusion
        from data_loaders.humanml_utils import get_inpainting_mask, BVH_JOINT_NAMES
        ee_names = ["R_Ankle", "L_Ankle", "L_Foot", "R_Foot"]



    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion
    if args.use_ddim:
        # model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion, timestep_respacing="ddim20")
        model, inpainting_diffusion, inpainting_diffusion_ddpm = creat_ddpm_ddim_diffusion(args, ModelClass=StyleDiffusion, timestep_respacing="ddim20")
    else:
        # model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion)
        model, inpainting_diffusion, inpainting_diffusion_ddpm = creat_ddpm_ddim_diffusion(args, ModelClass=StyleDiffusion)
    
    model.to(dist_util.dev())
    # for i, ((motion, cond), (normal_motion, normal_cond)) in enumerate(data):
    #     print(motion.shape, normal_motion.shape)
    model.eval()

    max_frames = 196 if args.dataset in ['kit', 'humanml', 'bandai-1_posrot', 'bandai-2_posrot'] else 60
    max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames  
    
    # input_motions, m_length = data.dataset.t2m_dataset.process_np_motion(path)
    # input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    # input_motions = input_motions.to(dist_util.dev()) 

    model_path = './save/my_motion_trajectory/model000050000.pt'
    trajectory_model = creat_motion_trajectory(args, data)
    trajectory_dict = torch.load(model_path, map_location='cpu')
    
    print("load trajectory prediction model: {}".format(model_path))
    load_model_wo_clip(trajectory_model, trajectory_dict)
    trajectory_model.to(dist_util.dev())
    trajectory_model.eval()

    input_motions, m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 
    
    with torch.no_grad():
        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1
        
        texts = ["predict root tarjectory"] * 1

        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs_style = collate(collate_args)    
        model_kwargs_style["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs_style["y"].items()}

        pred_root = trajectory_model(input_motions, **model_kwargs_style)
          
    pred_motion = torch.cat((pred_root, input_motions[:, 4:, ...]), 1)
    pred_motion = data.dataset.t2m_dataset.inv_transform(pred_motion.detach().cpu().permute(0, 2, 3, 1)).float()
    pred_motion = recover_from_ric(pred_motion, 22)
    pred_motion = pred_motion.view(-1, *pred_motion.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    pred_motion_np = pred_motion[0].transpose(2, 0, 1)[:m_length]
    pred_motion_np, _, _, _ = remove_fs("", pred_motion_np, pred_motion_np, BVH_JOINT_NAMES, ee_names, force_on_floor=False, interp_length=2, use_butterworth=True, use_vel3=True)
    
    path = pos2hmlrep(pred_motion_np)

    input_motions, m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 
    

    collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1
    if args.dataset == 'humanml':
        # caption = 'a figure is walking alone in place'
        # style_label = 'in fencing style'
        # caption = 'a person walks'
        caption = 'a person walks'
        style_label = args.style_file[:-4]

 
    texts = [caption] * 1
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)    
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())
    model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * 2.5

    # if args.use_ddim:
    #     sample_fn = inpainting_diffusion.ddim_sample_loop
    # else:
    #     sample_fn = inpainting_diffusion.p_sample_loop

    sample_fn = inpainting_diffusion_ddpm.p_sample_loop
    with torch.no_grad():
        if args.dataset == 'humanml':
            net = model.controlmdm
            # net = model.controlmdm.motion_enc.mdm_model
            sample = sample_fn(
            net,
            (1, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            # skip_timesteps = 50,  # 0 is the default value - i.e. don't skip any step
            # init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            )
    sample_t2m = sample.detach().clone()
    sample_t2m = data.dataset.t2m_dataset.inv_transform(sample_t2m.cpu().permute(0, 2, 3, 1)).float()
    sample_t2m = recover_from_ric(sample_t2m, 22)
    sample_t2m = sample_t2m.view(-1, *sample_t2m.shape[2:]).permute(0, 2, 3, 1)
    sample_t2m = sample_t2m.cpu().numpy()

    motion = sample_t2m[0].transpose(2, 0, 1)[:]
    fs_motion, _, _, _ = remove_fs("", motion, motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, use_window=False, use_vel3=True)

    rm_motion = pos2hmlrep(fs_motion)

    rm_motion, new_length = data.dataset.t2m_dataset.process_np_motion(rm_motion)
    rm_motion = torch.Tensor(rm_motion.T).unsqueeze(1).unsqueeze(0)
    rm_motion = rm_motion.to(dist_util.dev()) 
        
    
    save_file = 'style_transfer_content{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(args.save_dir, save_file)
    gt_frames_per_sample = {}
    plot_3d_motion(animation_save_path, skeleton, fs_motion, title=caption,
                dataset=args.dataset, fps=20, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
     

    model.train()
    data_style = ((rm_motion, model_kwargs), )    

    class InpaintingDataLoader(object):
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            for motion, cond in super().__getattribute__('data').__iter__():
                if args.inpainting_mask != "":
                    cond['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).to(motion.device)
                    # do not need inpainted motion when training
                    # cond['y']['inpainted_motion'] = motion.float()
                if args.dataset == 'humanml':
                    if args.weakly_style_pair:
                        for i in range(len(cond["y"]["tokens"])):
                            token_sent = cond["y"]["tokens"][i]
                            tokens = token_sent.split("_")
                            verb_idx = [i-1 for i, token in enumerate(tokens) if '/VERB' in token]
                            caption = cond['y']['text'][i].split(" ")
                            for j, idx in enumerate(verb_idx):
                                caption.insert(idx + 1 + j, style_label)
                            # caption.insert(0, style_label)
                            caption = " ".join(caption)
                            cond['y']['text'][i] = caption
                elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
                    if args.weakly_style_pair:
                        for i in range(len(cond["y"]["text"])):
                            caption = cond["y"]["text"][i].split(" ")
                            caption.pop(-1)
                            caption.insert(-1, style_label)
                            caption = " ".join(caption)
                            cond['y']['text'][i] = caption
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
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_enc()) / 1000000.0))
    print("Training...")
    InpaintingTrainLoop(args, train_platform, model, data, diffusion=inpainting_diffusion, style_data=data_style).run_loop()
    train_platform.close()

    model = model.eval()
    sample_fn = inpainting_diffusion.ddim_sample_loop
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())
    
    sample = sample_fn(
            model,
            sample.detach().shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps= int(700/1000 * 20),  # 0 is the default value - i.e. don't skip any step
            init_image=sample.detach(),
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn_with_grad=False,
            pred_xstart_in_graph=False,
            dump_all_xstart=False,
        )

 
    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        sample = sample.cpu().numpy() 

        
    motion = sample[0].transpose(2, 0, 1)[:]
    save_file = 'style_transfer_example{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(args.save_dir, save_file)
    gt_frames_per_sample = {}
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=20, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
    
    
    


if __name__ == "__main__":
    main()
