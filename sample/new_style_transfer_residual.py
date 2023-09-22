# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import reconstruct_args 
from utils.model_util import creat_diffusion_motionenc, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_array
import imageio
import shutil
from data_loaders.tensors import collate
from copy import deepcopy
from tqdm import tqdm


def main():
    args = reconstruct_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.motionenc_path))
    niter = os.path.basename(args.motionenc_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.motionenc_path),
                                'residual_stytrans_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data, sty_data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    diffusion, motion_enc = creat_diffusion_motionenc(args, data)

    print(f"Loading checkpoints from [{args.motionenc_path}]...")
    
    motion_enc_dict = torch.load(args.motionenc_path, map_location='cpu')
    load_model_wo_clip(motion_enc, motion_enc_dict)

    # if args.guidance_param != 1:
    #     mdm_model = ClassifierFreeSampleModel(mdm_model)   # wrapping model with the classifier-free sampler

    motion_enc.to(dist_util.dev())
    motion_enc.eval() # disable random masking

    if is_using_data:
        iterator = iter(data)
        t2m_motion, model_kwargs = next(iterator)
        t2m_motion = t2m_motion.to(dist_util.dev())
        model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}
        norm_motion, style_motion = GetBandaiExample(sty_data, n_frames)
        norm_motion = norm_motion.float().to(dist_util.dev())
        style_motion = style_motion.float().to(dist_util.dev())
        model_contkwargs = deepcopy(model_kwargs)
        model_stykwargs = deepcopy(model_kwargs)
        model_rawkwargs = deepcopy(model_kwargs)
        style_iterator = iter(sty_data)
        bandai_motion, model_bandaikwargs = next(style_iterator)
        bandai_motion = bandai_motion.to(dist_util.dev())
        model_bandaikwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}

    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    # add CFG scale to batch    
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param   
        model_contkwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_stykwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_bandaikwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_rawkwargs['y']['scale'] = torch.ones(args.batch_size,device=dist_util.dev()) * args.guidance_param                      
        

    norm_motion = motion_enc.mdm_model.re_encode(norm_motion)
    style_motion = motion_enc.mdm_model.re_encode(style_motion)
    content_mu, _ = motion_enc(norm_motion)
    style_mu, _= motion_enc(style_motion)
    t2m_mu, _= motion_enc(t2m_motion)
    
    # sty_mean = torch.mean(style_mu, -1, keepdim=True)
    # sty_std = torch.std(style_mu, -1, keepdim=True)
    # cont_mean = torch.mean(content_mu, -1, keepdim=True)
    # cont_std = torch.std(content_mu, -1, keepdim=True)
    # content_mu = (content_mu - cont_mean)/cont_std
    # content_mu = content_mu * sty_std + sty_mean
    residual = style_mu - content_mu

    model_contkwargs["mu"] = content_mu
    model_stykwargs["mu"] = style_mu
    model_kwargs["mu"] = t2m_mu + residual

    sample_fn = diffusion.p_sample_loop

    sample_t2m = sample_fn(
        motion_enc.mdm_model,
        (args.batch_size, motion_enc.mdm_model.njoints, motion_enc.mdm_model.nfeats, max_frames),  # BUG FIX
        clip_denoised=False,
        model_kwargs=model_rawkwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        
    )
    
    sample_cont = sample_fn(
            motion_enc.mdm_model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, motion_enc.mdm_model.njoints, motion_enc.mdm_model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_contkwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

    sample_sty = sample_fn(
            motion_enc.mdm_model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, motion_enc.mdm_model.njoints, motion_enc.mdm_model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_stykwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    
    sample_t2m_style = sample_fn(
            motion_enc.mdm_model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, motion_enc.mdm_model.njoints, motion_enc.mdm_model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    if motion_enc.mdm_model.data_rep == 'hml_vec':
        n_joints = 22 if sample_cont.shape[1] == 263 else 21
        
        sample_cont = data.dataset.t2m_dataset.inv_transform(sample_cont.cpu().permute(0, 2, 3, 1)).float()
        sample_cont = recover_from_ric(sample_cont, n_joints) # B 1 T J 3 
        sample_cont = sample_cont.view(-1, *sample_cont.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 

        sample_sty = data.dataset.t2m_dataset.inv_transform(sample_sty.cpu().permute(0, 2, 3, 1)).float()
        sample_sty = recover_from_ric(sample_sty, n_joints) # B 1 T J 3 
        sample_sty = sample_sty.view(-1, *sample_sty.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 
        
        sample_t2m_gt = data.dataset.t2m_dataset.inv_transform(t2m_motion.detach().cpu().permute(0, 2, 3, 1)).float()
        sample_t2m_gt = recover_from_ric(sample_t2m_gt, n_joints) # B 1 T J 3 
        sample_t2m_gt = sample_t2m_gt.view(-1, *sample_t2m_gt.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 
        
        sample_t2m_style = data.dataset.t2m_dataset.inv_transform(sample_t2m_style.detach().cpu().permute(0, 2, 3, 1)).float()
        sample_t2m_style = recover_from_ric(sample_t2m_style, n_joints) # B 1 T J 3 
        sample_t2m_style = sample_t2m_style.view(-1, *sample_t2m_style.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 
        
        sample_t2m = data.dataset.t2m_dataset.inv_transform(sample_t2m.cpu().permute(0, 2, 3, 1)).float()
        sample_t2m = recover_from_ric(sample_t2m, n_joints) # B 1 T J 3 
        sample_t2m = sample_t2m.view(-1, *sample_t2m.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 


    caption = model_kwargs['y']['text']
    # caption = model_kwargs["sty_y"]["text"]
    
    
    sample_cont = sample_cont.cpu().numpy().transpose(0, 3, 1, 2)
    sample_sty = sample_sty.cpu().numpy().transpose(0, 3, 1, 2)
        
    sample_t2m_gt = sample_t2m_gt.cpu().numpy().transpose(0, 3, 1, 2)
    sample_t2m_style = sample_t2m_style.cpu().numpy().transpose(0, 3, 1, 2)

    sample_t2m = sample_t2m.cpu().numpy().transpose(0, 3, 1, 2)
   

    cont_array = plot_3d_array([sample_cont[0][:n_frames], None, paramUtil.t2m_kinematic_chain, "content_motion_rec"])
    imageio.mimsave(os.path.join(out_path, 'reconstruct_content.gif'), np.array(cont_array), duration=6)

    sty_array = plot_3d_array([sample_sty[0][:n_frames], None, paramUtil.t2m_kinematic_chain, "stylized_motion_rec"])
    imageio.mimsave(os.path.join(out_path, 'reconstruct_style.gif'), np.array(sty_array), duration=6)
    for i in range(args.num_samples):
        t2m_source = plot_3d_array([sample_t2m_gt[i][:model_kwargs['y']["lengths"][i]], None, paramUtil.t2m_kinematic_chain, caption[i]])
        imageio.mimsave(os.path.join(out_path, f't2m_source_{i}.gif'), np.array(t2m_source), duration=model_kwargs['y']["lengths"][i]/20)
        
        t2m_pred = plot_3d_array([sample_t2m[i][:model_kwargs['y']["lengths"][i]], None, paramUtil.t2m_kinematic_chain, caption[i]])
        imageio.mimsave(os.path.join(out_path, f't2m_pred_{i}.gif'), np.array(t2m_pred), duration=model_kwargs['y']["lengths"][i]/20)

        t2m_style_trans = plot_3d_array([sample_t2m_style[i][:model_kwargs['y']["lengths"][i]], None, paramUtil.t2m_kinematic_chain, caption[i]+" style_trans"])
        imageio.mimsave(os.path.join(out_path, f't2m_style_trans_{i}.gif'), np.array(t2m_style_trans), duration=model_kwargs['y']["lengths"][i]/20)

def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='eval')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames

    sty_data = get_dataset_loader(name=args.style_dataset,
                                  batch_size=args.batch_size,
                                  num_frames=max_frames, 
                                  split='test',
                                  hml_mode="text_only"
                                    )
    return data, sty_data

def GetBandaiExample(sty_loader, n_frames):
    # norm_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_raise-up-both-hands_normal_024.npy")
    norm_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_walk_normal_013.npy")
    # style_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_raise-up-both-hands_feminine_014.npy")
    # style_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_raise-up-both-hands_active_033.npy")
    # style_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_walk_active_017.npy")
    # style_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_walk_elderly_009.npy")
    style_npy = os.path.join(sty_loader.dataset.opt.motion_dir, "dataset-2_walk_feminine_017.npy")
    
    norm_motion = np.load(norm_npy)
    style_motion = np.load(style_npy)
    
    norm_len = norm_motion.shape[0]
    style_len = style_motion.shape[0]
    
    norm_motion = (norm_motion - sty_loader.dataset.style_dataset.mean) / sty_loader.dataset.style_dataset.std
    style_motion = (style_motion - sty_loader.dataset.style_dataset.mean) / sty_loader.dataset.style_dataset.std

    if norm_len < n_frames:
        norm_motion = np.concatenate([norm_motion,
                                    np.zeros((n_frames - norm_len, norm_motion.shape[1]))
                                    ], axis=0)
    else:
        norm_motion = norm_motion[:n_frames]
        
    if style_len < n_frames:
        style_motion = np.concatenate([style_motion,
                                    np.zeros((n_frames - style_len, style_motion.shape[1]))
                                    ], axis=0)
    else:
        style_motion = style_motion[:n_frames]
    
    norm_motion = torch.from_numpy(norm_motion).T.unsqueeze(1).unsqueeze(0)
    style_motion = torch.from_numpy(style_motion).T.unsqueeze(1).unsqueeze(0)
    
    return norm_motion, style_motion
    

if __name__ == "__main__":
    main()
