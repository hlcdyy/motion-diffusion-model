# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import json
from utils.fixseed import fixseed
from utils.parser_util import eval_inpainting_style_args, predict_trajectory_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_stylediffuse_and_diffusion, load_model_wo_controlmdm, creat_serval_diffusion,load_model_wo_moenc, creat_motion_trajectory, load_model_wo_clip
import torch 
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
import data_loaders.humanml.utils.paramUtil as paramUtil
import shutil
import numpy as np
from model.cfg_sampler import ClassifierFreeSampleModel
from scipy import io
from data_loaders.humanml.common.bvh_utils import output_bvh_from_real_rot, remove_fs, fit_joints_bvh, fit_joints_bvh_quats
from visualize.vis_utils import joints2rotation, joints2bvh
from data_loaders.humanml.common.skeleton import Skeleton


def main():
    args = eval_inpainting_style_args()
    output_bvh = True
    use_dataset = False
    use_eval_sample = True
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml',"bandai-1_posrot", "bandai-2_posrot"] else 60
    max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.input_text != '':
            out_path += '_' + args.input_text.replace(' ', '_').replace('.', '')


    print("creating data loader...")
    args.batch_size = args.num_samples
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
        from model.mdm_forstyledataset import StyleDiffusion
        if args.dataset == 'stylexia_posrot':
            from data_loaders.stylexia_posrot_utils import get_inpainting_mask
            data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='test')
        else:
            from data_loaders.bandai_posrot_utils import get_inpainting_mask
            data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='all')
        
    else:
        from data_loaders.humanml_utils import get_inpainting_mask, BVH_JOINT_NAMES
        from model.mdm import StyleDiffusion
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='test', shuffle=True)
        if use_dataset:
            iteration = iter(data)

        skeleton = paramUtil.t2m_kinematic_chain
        real_offset = paramUtil.smpl_real_offsets
        ee_names = ["R_Ankle", "L_Ankle", "L_Foot", "R_Foot"]
        anim = Skeleton(torch.Tensor(paramUtil.smpl_raw_offsets), skeleton, dist_util.dev())

        print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion

    use_ddim = args.use_ddim

    if use_ddim:
        model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion, timestep_respacing="ddim20")
    else:
        model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion)
    
    inpainting_style_dict = torch.load(args.model_path, map_location='cpu')
    print("load style diffusion model: {}".format(args.model_path))
    if args.dataset == 'humanml':
        load_model_wo_controlmdm(model, inpainting_style_dict)
    else:
        load_model_wo_moenc(model, inpainting_style_dict)

    model.to(dist_util.dev())
    model.eval()

    args.model_path = './save/my_motion_trajectory/model000050000.pt'
    trajectory_model = creat_motion_trajectory(args, data)
    trajectory_dict = torch.load(args.model_path, map_location='cpu')
    
    print("load trajectory prediction model: {}".format(args.model_path))
    load_model_wo_clip(trajectory_model, trajectory_dict)
    trajectory_model.to(dist_util.dev())
    trajectory_model.eval()

    # fill this
    # motion_dir = '/home/synthesis/xuyiwen/TrackDnD/datasets/3DPW/estimated_v2/outdoors_fencing_01'
    # path = os.path.join(motion_dir, 'hybrik.pt')
    # motion_dir = './videos/res_xixiyu/'
    # motion_dir = './videos/res_xixiyu_glamr_static/pose_est'
    # motion_dir = './videos/H36M_S6/Hybrik_Walking_1.60457274/'
    # motion_dir = './videos/100_walks/'
    # motion_dir = '/mnt/data_0/hulei/StyleProjects/sig_video/mocap_0/'
    motion_dir = '/mnt/data_0/hulei/StyleProjects/sig_video/mocap_1/'

    path = os.path.join(motion_dir, 'res.pk')
    # path = os.path.join(motion_dir, 'pose.pkl')
    # trans_dir = './videos/res_xixiyu_glamr_static/pose_est'
    # trans_path = os.path.join(trans_dir, 'pose.pkl')
    # fps_video = 30 # xixiyu
    # fps_video = 24 # 100 walk
    # fps_video = 50 #/H36M_S6/Hybrik_Walking_1.60457274/
    fps_video = 25 # mocap_0, 1

    # start_second = 52.4 # xixiyu 潜行
    # end_second = 55.5
    # start_second = 15
    # end_second = 22
    # start_second = 2.8 # 牛仔
    # end_second = 5
    # start_second = 56 # xixiyu tired
    # end_second = 61
    # start_second = 39.4 # xixiyu 逆风
    # end_second = 44
    # start_second = 10 
    # end_second = 13
    # start_second = 33
    # end_second = 36
    # start_second = 14
    # end_second = 17.4
    # start_second = 56
    # end_second = 58 # old man
    # start_second = 70 # Mummy
    # end_second = 73.5
    # start_second = 101 # sore back
    # end_second = 103
    start_second = 0.1 # run happily
    end_second = 3
    
        
    from utils.process_smpl_from_hybrik import amass_to_pose, pos2hmlrep
    pose_seq_np_n, joints = amass_to_pose(path, fps_video, trans_path="", with_trans=False)
    path = pos2hmlrep(joints)
    
    end_frame = int(20*end_second) if int(20*end_second) < path.shape[0] else path.shape[0]
    path = path[int(20*start_second):end_frame] # after process fixed 20 fps
    
    # motion_dir = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs'
    # path = os.path.join(motion_dir, 'M008551.npy')
    
    if not use_dataset:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
        

    input_motions, style_m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 
    
    with torch.no_grad():
        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': style_m_length}] * 1
        
        texts = ["predict root tarjectory"] * args.num_samples

        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs_style = collate(collate_args)    
        model_kwargs_style["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs_style["y"].items()}
            
        pred_root = trajectory_model(input_motions, **model_kwargs_style)
          
    pred_motion = torch.cat((pred_root, input_motions[:, 4:, ...]), 1)
    pred_motion = data.dataset.t2m_dataset.inv_transform(pred_motion.detach().cpu().permute(0, 2, 3, 1)).float()
    pred_motion = recover_from_ric(pred_motion, 22)
    pred_motion = pred_motion.view(-1, *pred_motion.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    pred_motion_np = pred_motion[0].transpose(2, 0, 1)[:style_m_length]
    path = pos2hmlrep(pred_motion_np)

    input_motions, style_m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 
    
    
    if args.dataset == 'humanml':
        if not use_dataset and not use_eval_sample:  
            collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': max_frames}] * 1
            if args.input_text != '':
                texts = [args.input_text] * args.num_samples

            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            _, model_kwargs = collate(collate_args)    
            model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * 2.5

            # model_kwargs['y']['inpainted_motion'] = input_motions
            # model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())

            # sample_fn = normal_diffusion.ddim_sample_loop
            sample_fn = normal_diffusion.p_sample_loop
            # sample_fn = inpainting_diffusion.ddim_sample_loop
            
            sample_t2m = sample_fn(
                model.controlmdm.motion_enc.mdm_model,
                # model.controlmdm,
                (1, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=10,  # 0 is the default value - i.e. don't skip any step
                init_image=input_motions,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        elif use_eval_sample:
            # fill this
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/014305.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/000299.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/001520.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/008583.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/007209.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/010996.npy'
            sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/010817.npy'

            sample_t2m, sample_length = data.dataset.t2m_dataset.process_np_motion(sample_path)
            sample_t2m = torch.Tensor(sample_t2m.T).unsqueeze(1).unsqueeze(0)
            sample_t2m = sample_t2m.to(dist_util.dev())
            
            collate_args = [{'inp': sample_t2m, 'tokens': None, 'lengths': sample_length}] * 1

            # texts = ['man runs left from the middle, then all the way right, then back left and stops at the middle'] 
            # texts = ['person is walking straight backwards']
            # texts = ['a man walks forward two steps and kicks his right leg and foot into the air two times in a row. he then walks backwards two steps and pauses, followed by walking forward two steps and kicks into the air with his left leg and foot.']
            # texts = ['person walks in an s shape']
            # texts = ['this person waves widly with his left hand']
            # texts = ['a man standing on one leg and moves the other around']
            texts = ['a person walks three steps to his right, then five steps to his left, and finally three steps to his right']

            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            _, model_kwargs = collate(collate_args)

            out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
            out_path += sample_path.split('/')[-1][:-4]
            if not os.path.exists(out_path):
                os.makedirs(out_path)

        else:
            sample_t2m, model_kwargs = next(iteration)
            sample_t2m = sample_t2m.to(dist_util.dev())
            
            out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
            out_path += '_results_from_testset'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

        # sample_t2m[:, :3, ...] = input_motions[:, :3, ...]
    elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
        if args.dataset == "stylexia_posrot":
            path = os.path.join(motion_dir, '029neutral_normal walking.npy')
        else:
            path = os.path.join(motion_dir, 'dataset-2_wave-left-hand_normal_001.npy')
        
        sample_t2m, m_length = data.dataset.t2m_dataset.process_np_motion(path)
        sample_t2m = torch.Tensor(sample_t2m.T).unsqueeze(1).unsqueeze(0)
        sample_t2m = sample_t2m.to(dist_util.dev())

        if args.input_text != '':
            texts = [args.input_text] * args.num_samples
        else:
            texts = ["A person walks normal"] * args.num_sample
        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1 # m_length
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)
  
    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample_t2m.shape[1] == 263 else 21
        sample_t2m_denorm = data.dataset.t2m_dataset.inv_transform(sample_t2m.cpu().permute(0, 2, 3, 1)).float()
        sample_t2m_np = recover_from_ric(sample_t2m_denorm, n_joints)
        sample_t2m_np = sample_t2m_np.view(-1, *sample_t2m_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

        save_bvh = 'input_content_motion.bvh'
        bvh_save_path = os.path.join(out_path, save_bvh)
        # output_bvh_from_real_rot(bvh_save_path, sample_t2m_np[0][0][:m_length], joint_num, skeleton, real_offset, BVH_JOINT_NAMES)

        if use_eval_sample:
            ref_motion = sample_t2m_np[0].transpose(2, 0, 1)[:sample_length]
        else:
            ref_motion = sample_t2m_np[0].transpose(2, 0, 1)[:max_frames]
        if args.dataset == 'humanml':
            ref_motion, _, _, _ = remove_fs("", ref_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, interp_length=2, use_vel3=True,after_butterworth=True)
            if output_bvh:
                joints2bvh(bvh_save_path, ref_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=100)
        
    # from scipy import io
    # io.savemat(os.path.join(out_path, "pred_xstart.mat"), {"x0": sample_t2m_np})

    mask = model_kwargs['y']['mask']

    all_lengths = []
    all_motions = []
    all_text = []

    # sample_t2m_xz = sample_t2m[:, 0:3, ...].clone()
    # sample_t2m[:, 0:3, ...] = input_motions[:, 0:3, :, :]
    model_kwargs['y']['inpainted_motion'] = sample_t2m 
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, sample_t2m.shape)).float().to(dist_util.dev())
    # args.inpainting_mask
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')
        
        if use_ddim:
            sample_fn = inpainting_diffusion.ddim_sample_loop
            skip_timesteps = int(args.skip_steps/args.diffusion_steps * 20)
        else:
            sample_fn = inpainting_diffusion.p_sample_loop
            skip_timesteps = args.skip_steps

        sample = sample_fn(
            model,
            sample_t2m.shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=sample_t2m,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn_with_grad=False,
            pred_xstart_in_graph=False,
            dump_all_xstart=True,
        )
        sample = sample[-3]
        # sample[:, 0:3, ...] = sample_t2m_xz

        # my_t = torch.ones([sample.shape[0]], device=dist_util.dev(), dtype=torch.long) * 14
        # noise_transfer = torch.rand_like(sample)
        # x_t = inpainting_diffusion.q_sample(sample, my_t, noise=noise_transfer, model_kwargs=model_kwargs)
        
        # sample = model.controlmdm(x_t, inpainting_diffusion._scale_timesteps(my_t), **model_kwargs)

        sample = sample_fn(
            model.controlmdm,
            sample.shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=int(300/args.diffusion_steps * 20),  # 0 is the default value - i.e. don't skip any step
            init_image=sample,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            dump_all_xstart=True,
            )
        sample = sample[-3]

    
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    
    gt_frames_per_sample = {}
    
    npy_path = os.path.join(out_path, 'results.npy')
    mat_path = os.path.join(out_path, 'results.mat')
    
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")

    if args.dataset == 'kit':
        skeleton = paramUtil.kit_kinematic_chain
    elif args.dataset == 'humanml':
        skeleton = paramUtil.t2m_kinematic_chain
    elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        skeleton = paramUtil.bandai_kinematic_chain
    elif args.dataset == 'stylexia_posrot':
        skeleton = paramUtil.xia_kinematic_chain
        
    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions_denorm = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions_denorm, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

        save_bvh = 'input_style_motion.bvh'
        bvh_save_path = os.path.join(out_path, save_bvh)
        sty_motion = input_motions[0].transpose(2, 0, 1)[:style_m_length]
        if output_bvh:
            joints2bvh(bvh_save_path, sty_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=100)

    length = all_lengths[0]
    fs_motion = all_motions[0].transpose(2, 0, 1)[:length].copy()
    # fs_motion, foot_vels, contacts, butter_motion = remove_fs("", fs_motion, fs_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, use_window=True)
    # fs_motion, foot_vels, contacts, butter_motion = remove_fs("", fs_motion, fs_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False,use_butterworth=True,use_vel3=True, vel3_thr=0.02)
    fs_motion, _, _, _ = remove_fs("", fs_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=True, use_butterworth=True,use_vel3=True, vel3_thr=0.03, after_butterworth=True)
    
    fs_motion, _, _, _ = remove_fs("", fs_motion, fs_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=True,use_butterworth=True,use_vel3=True, vel3_thr=0.04,  after_butterworth=True, interp_length=3)

    save_bvh = 'out_transferred_motion.bvh'
    bvh_save_path = os.path.join(out_path, save_bvh)
    if output_bvh and not use_dataset:
        joints2bvh(bvh_save_path, fs_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=100, Butterworth_all=True)

        new_index = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
        fit_joints_bvh_quats(bvh_save_path, real_offset, fs_motion[..., new_index, :])


    print(f"saving results file to [{npy_path}]")
    length = model_kwargs['y']['lengths'][0]
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    io.savemat(mat_path,
               {"motion": fs_motion, 
                "content_motion": sample_t2m_np[0].transpose(2, 0, 1)[:length],
                "style_motion": input_motions[0].transpose(2, 0, 1)[:style_m_length],
                'text': all_text, 'lengths': all_lengths,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})


    sample_i = 0
    rep_files = []
    caption = 'Input Content Motion'
    if use_dataset:
        for i in range(args.batch_size):
            length = model_kwargs['y']['lengths'][i]
            caption = model_kwargs['y']['text'][i]
            motion = sample_t2m_np[i].transpose(2, 0, 1)[:length]
            save_file = 'input_content_motion_{}.mp4'.format(model_kwargs['y']['file_name'][i])
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({0}) "{caption}" | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode='gt',
                        gt_frames=gt_frames_per_sample.get(0, []))
    else:
        length = model_kwargs['y']['lengths'][0]
        motion = sample_t2m_np[0].transpose(2, 0, 1)[:length]
        save_file = 'input_content_motion{:02d}.mp4'.format(0)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files.append(animation_save_path)
        print(f'[({0}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                    dataset=args.dataset, fps=fps, vis_mode='root_horizontal',
                    gt_frames=gt_frames_per_sample.get(0, []), painting_features="root_horizontal".split(","))
    
    caption = 'Input Style Motion'
    motion = input_motions[0].transpose(2, 0, 1)[:style_m_length]
    save_file = 'input_style_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []), painting_features="root_horizontal".split(","))
    
    if use_dataset:
        total_i = args.batch_size
    else:
        total_i = 1
    for sample_i in range(total_i):

        for rep_i in range(args.num_repetitions):
            
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]

            fs_motion = motion.copy()
            c_motion = sample_t2m_np[sample_i].transpose(2, 0, 1)[:length]

            fs_motion, _, _, _ = remove_fs("", fs_motion, c_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False,use_butterworth=True,use_vel3=True, vel3_thr=0.02, after_butterworth=True)
            motion = fs_motion

            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            if use_dataset:
                save_file = 'sample{}_rep{:02d}.mp4'.format(model_kwargs['y']['file_name'][sample_i], rep_i)

                if output_bvh:
                    bvh_save_path = os.path.join(out_path, save_file.replace('.mp4', '.bvh'))
                    joints2bvh(bvh_save_path, motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=30)

            
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset=args.dataset, fps=fps, vis_mode="root_horizontal",
                            gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features="root_horizontal".split(","))
        
            
            if args.num_repetitions > 1:
                all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
                ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
                hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + 1} '
                ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
                os.system(ffmpeg_rep_cmd)
                print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
            
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')
   

if __name__ == "__main__":
    main()
