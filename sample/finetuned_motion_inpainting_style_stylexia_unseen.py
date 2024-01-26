# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
from utils.fixseed import fixseed
from utils.parser_util import eval_inpainting_style_args
from utils import dist_util
from train.training_loop import TrainInpaintingLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_stylediffuse_and_diffusion, load_model_wo_controlmdm, creat_serval_diffusion,load_model_wo_moenc
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch 
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
import data_loaders.humanml.utils.paramUtil as paramUtil
import shutil
import numpy as np
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.bvh_utils import output_bvh_from_real_rot, remove_fs, fit_joints_bvh, fit_joints_bvh_quats
from visualize.vis_utils import joints2rotation, joints2bvh
from scipy import io

def main():
    output_bvh = True
    use_dataset = False
    use_eval_sample = True
    args = eval_inpainting_style_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml',"bandai-1_posrot", "bandai-2_posrot"] else 60
    max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    
    args.inpainting_mask = 'right_hand,root_horizontal'
    
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.input_text != '':
            out_path += '_' + args.input_text.replace(' ', '_').replace('.', '')


    print("creating data loader...")
    # args.batch_size = args.num_samples
   
    from data_loaders.humanml_utils import get_inpainting_mask, BVH_JOINT_NAMES
    from model.mdm import StyleDiffusion
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='test')
    
    if use_dataset:
        iteration = iter(data)
    # iteration = iter(data)
    # sample_t2m, model_kwargs = next(iteration)
    # sample_t2m = sample_t2m.to(dist_util.dev())
    # model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}    

    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion

    use_ddim = True

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

    if args.dataset == 'humanml':
     
        motion_dir = '/data/hulei/Projects/Style100_2_HumanML/style_xia_humanml/smpl_joint_vecs'
        if not args.style_file:
            raise RuntimeError
        path = os.path.join(motion_dir,args.style_file)
        joint_num = 22

    if args.dataset == 'humanml':
        skeleton = paramUtil.t2m_kinematic_chain
        real_offset = paramUtil.smpl_real_offsets
        ee_names = ["R_Ankle", "L_Ankle", "L_Foot", "R_Foot"]
        anim = Skeleton(torch.Tensor(paramUtil.smpl_raw_offsets), skeleton, dist_util.dev())

    if not use_dataset:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
    
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
            
            sample_fn = normal_diffusion.p_sample_loop
            # sample_fn = normal_diffusion.ddim_sample_loop
            # sample_fn = inpainting_diffusion.ddim_sample_loop

            sample_t2m = sample_fn(
                model.controlmdm.motion_enc.mdm_model,
                (1, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                # init_image=input_motions,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        elif use_eval_sample:
            # fill the sample_path and texts
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/000299.npy'
            sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/006724.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/001059.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/008967.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/001520.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/009041.npy'
            # sample_path = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs/008583.npy'

            sample_t2m, sample_length = data.dataset.t2m_dataset.process_np_motion(sample_path)
            sample_t2m = torch.Tensor(sample_t2m.T).unsqueeze(1).unsqueeze(0)
            sample_t2m = sample_t2m.to(dist_util.dev())
            
            collate_args = [{'inp': sample_t2m, 'tokens': None, 'lengths': sample_length}] * 1

            # texts = ['person is walking straight backwards'] * 1
            texts = ['a person is walking around in circles with their right arm resting on or in something']
            # texts = ['the person is holding their head while walking']
            # texts = ['the person is playing dodgeball']
            # texts = ['a man walks forward two steps and kicks his right leg and foot into the air two times in a row. he then walks backwards two steps and pauses, followed by walking forward two steps and kicks into the air with his left leg and foot.']
            # texts = ['a person does a throwing motion with his right arm']
            # texts = ['person walking in an s shape']

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
        sample_t2m_denorm = data.dataset.t2m_dataset.inv_transform(sample_t2m.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
        sample_t2m_np = recover_from_ric(sample_t2m_denorm, joint_num)
        sample_t2m_np = sample_t2m_np.view(-1, *sample_t2m_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        #  B J 3 T
    
        save_bvh = 'input_content_motion.bvh'
        bvh_save_path = os.path.join(out_path, save_bvh)
        # output_bvh_from_real_rot(bvh_save_path, sample_t2m_np[0][0][:m_length], joint_num, skeleton, real_offset, BVH_JOINT_NAMES)

        if use_eval_sample:
            ref_motion = sample_t2m_np[0].transpose(2, 0, 1)[:sample_length]
        else:
            ref_motion = sample_t2m_np[0].transpose(2, 0, 1)[:max_frames]

        if args.dataset == 'humanml':
            ref_motion, _, _, _ = remove_fs("", ref_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, use_butterworth=True, use_vel3=True, vel3_thr=0.01, interp_length=1, after_butterworth=True)
            content_rm_motion = ref_motion
            
        if args.dataset != 'humanml':
            fit_joints_bvh(bvh_save_path, sample_t2m_denorm[0, 0, :m_length, :], joint_num, anim, real_offset, ref_motion, BVH_JOINT_NAMES)
        else:
            if output_bvh and not use_dataset:
                joints2bvh(bvh_save_path, ref_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=200)
        
    # from scipy import io
    # io.savemat(os.path.join(out_path, "pred_xstart.mat"), {"x0": sample_t2m_np})

    mask = model_kwargs['y']['mask']

    all_lengths = []
    all_motions = []
    all_text = []
    all_hml_motions = []

    model_kwargs['y']['inpainted_motion'] = sample_t2m 
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, sample_t2m.shape)).float().to(dist_util.dev())
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')
        
        if use_ddim:
            sample_fn = inpainting_diffusion.ddim_sample_loop
            skip_timesteps = int(args.skip_steps/args.diffusion_steps * 20)
        else:
            sample_fn = inpainting_diffusion.p_sample_loop
            skip_timesteps = args.skip_steps
        
        if args.dataset == 'humanml':
            dump_all_xstart = True
        else:
            dump_all_xstart = True
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
            dump_all_xstart=dump_all_xstart,
        )
        if dump_all_xstart:
            sample = sample[-3]
            # sample = sample[-4]
        else:
            sample = sample

        if model.data_rep == 'hml_vec':
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            all_hml_motions.append(sample)
            sample = recover_from_ric(sample, joint_num)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)
    all_hml_motions = torch.cat(all_hml_motions, dim=0)

    
    gt_frames_per_sample = {}
    
    npy_path = os.path.join(out_path, 'results.npy') 
    mat_path = os.path.join(out_path, 'results.mat')

    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))


    print(f"saving visualizations to [{out_path}]...")

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions_denorm = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions_denorm, joint_num)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        
        save_bvh = 'input_style_motion.bvh'
        bvh_save_path = os.path.join(out_path, save_bvh)
        # output_bvh_from_real_rot(bvh_save_path, input_motions[0][0][:style_m_length], joint_num, skeleton, real_offset, BVH_JOINT_NAMES)
        sty_motion = input_motions[0].transpose(2, 0, 1)[:style_m_length]

        if args.dataset != 'humanml':
            fit_joints_bvh(bvh_save_path, input_motions_denorm[0, 0, :style_m_length], joint_num, anim, real_offset, sty_motion, BVH_JOINT_NAMES)
        else:
            if output_bvh:
                joints2bvh(bvh_save_path, sty_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=150)


    length = all_lengths[0]
    fs_motion = all_motions[0].transpose(2, 0, 1)[:length].copy()
    # fs_motion, foot_vels, contacts, butter_motion = remove_fs("", fs_motion, fs_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, use_vel3=True, vel3_thr=0.03, after_butterworth=True)
    fs_motion, foot_vels, contacts, butter_motion = remove_fs("", fs_motion, content_rm_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, use_vel3=True, vel3_thr=0.02, use_butterworth=True, after_butterworth=True)
    # rotations = joints2rotation(fs_motion, 150) # [1, 25, 6, length] 24 joint rota + root trans

    save_bvh = 'out_transferred_motion.bvh'
    bvh_save_path = os.path.join(out_path, save_bvh)
    if args.dataset != 'humanml':
        fit_joints_bvh(bvh_save_path, all_hml_motions[0, 0, :length], joint_num, anim, real_offset, fs_motion, BVH_JOINT_NAMES)
    else:
        if output_bvh and not use_dataset:
            joints2bvh(bvh_save_path, fs_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=200)

            new_index = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
            fit_joints_bvh_quats(bvh_save_path, real_offset, fs_motion[..., new_index, :])


    
   
    io.savemat(mat_path,
               {"motion": all_motions.transpose(3, 1, 2, 0), 
                "content_motion": sample_t2m_np[0].transpose(2, 0, 1)[:length],
                "style_motion": input_motions[0].transpose(2, 0, 1)[:style_m_length],
                'text': all_text, 'lengths': all_lengths,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
                "foot_vels": foot_vels, "rmfs_motion":fs_motion, "butter_motion": butter_motion,
                "foot_contacts": contacts})


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
            
            # if output_bvh:
            #     bvh_save_path = os.path.join(out_path, save_file.replace('.mp4', '.bvh'))
            #     joints2bvh(bvh_save_path, motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=30)

            rep_files.append(animation_save_path)
            print(f'[({0}) "{caption}" | -> {save_file}]')
            try:
                plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset=args.dataset, fps=fps, vis_mode='gt',
                            gt_frames=gt_frames_per_sample.get(0, []))
            except:
                continue
    else:
        length = model_kwargs['y']['lengths'][0]
        motion = sample_t2m_np[0].transpose(2, 0, 1)[:length]
        save_file = 'input_content_motion{:02d}.mp4'.format(0)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files.append(animation_save_path)
        print(f'[({0}) "{caption}" | -> {save_file}]')
        try:
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode='gt',
                        gt_frames=gt_frames_per_sample.get(0, []))
        except:
            print("skip")
    
    caption = 'Input Style Motion'
    motion = input_motions[0].transpose(2, 0, 1)[:style_m_length]
    save_file = 'input_style_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    try:
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                    dataset=args.dataset, fps=fps, vis_mode='gt',
                    gt_frames=gt_frames_per_sample.get(0, []))
    except:
        print("skip")
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
            if use_dataset:
                motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                fs_motion = motion.copy()
                c_motion = sample_t2m_np[sample_i].transpose(2, 0, 1)[:length]

                fs_motion, _, _, _ = remove_fs("", fs_motion, c_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False,use_butterworth=True,use_vel3=True, vel3_thr=0.02)
                motion = fs_motion
                
            else:
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
            try:
                plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                                dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                                gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','))
            except:
                print("skip")
                
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
