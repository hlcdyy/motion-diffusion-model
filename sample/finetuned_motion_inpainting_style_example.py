# This code is based on https://github.com/openai/guided-diffusion


# this script is used for visualize noised motion in different steps
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
from utils.fixseed import fixseed
from utils.parser_util import eval_inpainting_style_args
from utils import dist_util
from train.training_loop import TrainInpaintingLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_stylediffuse_and_diffusion, load_model_wo_controlmdm, creat_serval_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from model.mdm import StyleDiffusion
from data_loaders.humanml_utils import get_inpainting_mask
import torch 
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
import data_loaders.humanml.utils.paramUtil as paramUtil
import shutil
import numpy as np


def main():
    args = eval_inpainting_style_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.input_text != '':
            out_path += '_' + args.input_text.replace(' ', '_').replace('.', '')


    print("creating data loader...")
    args.batch_size = args.num_samples
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='test')
    
    # debug
    # iteration = iter(data)
    # motion, cond = next(iteration)
    
    # for i in range(len(cond["y"]["tokens"])):
    #     token_sent = cond["y"]["tokens"][i]
    #     tokens = token_sent.split("_")
    #     verb_idx = [i-1 for i, token in enumerate(tokens) if '/VERB' in token]
    #     caption = cond['y']['text'][i].split(" ")
    #     for j, idx in enumerate(verb_idx):
    #         caption.insert(idx + 1 + j, 'happliy')
    #     caption = " ".join(caption)
    #     cond['y']['text'][i] = caption

    # debug

    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion
    model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion)
    inpainting_style_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_controlmdm(model, inpainting_style_dict)
    model.to(dist_util.dev())

    model.eval()

    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    motion_dir = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs'
    path = os.path.join(motion_dir, 'M008551.npy')
    
    input_motions, m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 

    collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1
    caption = 'a figure skips in a circle'

    if args.input_text != '':
        texts = [args.input_text] * args.num_samples

    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)    
    model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * 2.5

    sample_fn = normal_diffusion.p_sample_loop

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
    
    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample_t2m.shape[1] == 263 else 21
        sample_t2m_np = data.dataset.t2m_dataset.inv_transform(sample_t2m.cpu().permute(0, 2, 3, 1)).float()
        sample_t2m_np = recover_from_ric(sample_t2m_np, n_joints)
        sample_t2m_np = sample_t2m_np.view(-1, *sample_t2m_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

    img = torch.randn(*sample_t2m.shape, device=dist_util.dev())
    my_t = torch.ones([sample_t2m.shape[0]], device=dist_util.dev(), dtype=torch.long) * 600
    noised_img = normal_diffusion.q_sample(sample_t2m, my_t, img, model_kwargs=model_kwargs)
    
    if model.data_rep == 'hml_vec':
        n_joints = 22 if noised_img.shape[1] == 263 else 21
        noised_img_np = data.dataset.t2m_dataset.inv_transform(noised_img.cpu().permute(0, 2, 3, 1)).float()
        noised_img_np = recover_from_ric(noised_img_np, n_joints)
        noised_img_np_t2m = noised_img_np.view(-1, *noised_img_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    
    mask = model_kwargs['y']['mask']

    all_lengths = []
    all_motions = []
    all_text = []

    model_kwargs['y']['inpainted_motion'] = input_motions 
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())

    img = torch.randn(*input_motions.shape, device=dist_util.dev())
    my_t = torch.ones([input_motions.shape[0]], device=dist_util.dev(), dtype=torch.long) * 600
    noised_img = inpainting_diffusion.q_sample(input_motions, my_t, img, model_kwargs=model_kwargs)
    # noised_img 的轨迹确实会变，因为x_start会乘一个系数
    
    if model.data_rep == 'hml_vec':
        n_joints = 22 if noised_img.shape[1] == 263 else 21
        noised_img_np = data.dataset.t2m_dataset.inv_transform(noised_img.cpu().permute(0, 2, 3, 1)).float()
        noised_img_np = recover_from_ric(noised_img_np, n_joints)
        noised_img_np = noised_img_np.view(-1, *noised_img_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    
    # del model_kwargs['y']['inpainted_motion']  
    # del model_kwargs['y']['inpainting_mask'] 
    
    # model_kwargs['y']['inpainting_mask'] = torch.zeros_like(model_kwargs['y']['inpainting_mask']).float().to(dist_util.dev())
    
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        sample_fn = inpainting_diffusion.p_sample_loop
        # sample_fn = normal_diffusion.p_sample_loop
        
        sample = sample_fn(
            model.controlmdm,
            # model.controlmdm.motion_enc.mdm_model,
            input_motions.shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
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

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    gt_frames_per_sample = {}
    
    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")

    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

    
    # model_kwargs['y']['inpainted_motion'] = input_motions 
    # model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())

    sample_i = 0
    rep_files = []
    caption = 'Noised Style Motion'
    length = model_kwargs['y']['lengths'][0]
    motion = noised_img_np[0].transpose(2, 0, 1)[:length]
    save_file = 'Noised Style Motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
    
    caption = 'Noised T2M Motion'
    length = model_kwargs['y']['lengths'][0]
    motion = noised_img_np_t2m[0].transpose(2, 0, 1)[:length]
    save_file = 'noised_content_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))

    caption = 'T2M Motion'
    length = model_kwargs['y']['lengths'][0]
    motion = sample_t2m_np[0].transpose(2, 0, 1)[:length]
    save_file = 'Content_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
        

    caption = 'Input Style Motion'
    length = model_kwargs['y']['inpainted_motion'].shape[-1]
    motion = input_motions[0].transpose(2, 0, 1)[:length]
    save_file = 'input_style_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))


    for rep_i in range(args.num_repetitions):
        
        caption = all_text[rep_i*args.batch_size + sample_i]
        if args.guidance_param == 0:
            caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
        else:
            caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
        length = all_lengths[rep_i*args.batch_size + sample_i]
        motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
        save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files.append(animation_save_path)
        print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                        gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','))
        
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
