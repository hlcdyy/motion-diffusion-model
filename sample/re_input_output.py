# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import style_transfer_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
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



def main():
    args = style_transfer_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'styletransfer_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

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
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        sty_iterator = iter(sty_data)
        _, model_kwargs = next(iterator)
        sty_motion, sty_kwargs = next(sty_iterator)
        model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}
        model_rawkwargs = deepcopy(model_kwargs)
        model_kwargs["sty_x"] = sty_motion.to(dist_util.dev())
        sty_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in sty_kwargs["y"].items()}
        model_kwargs["sty_y"] = sty_kwargs["y"]
        
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
    all_sty_motions = []
    all_sty_names = []
    all_motions_wosty =[]

    # add CFG scale to batch    
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_rawkwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    
    sample_fn = diffusion.p_sample_loop
    sample_wosty = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_rawkwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    re_motion = model.model.re_encode(sample_wosty)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    if model.data_rep == 'hml_vec':
        n_joints = 22 if re_motion.shape[1] == 263 else 21
            
        sample_wosty = data.dataset.t2m_dataset.inv_transform(sample_wosty.cpu().permute(0, 2, 3, 1)).float()
        sample_wosty = recover_from_ric(sample_wosty, n_joints) # B 1 T J 3 
        sample_wosty = sample_wosty.view(-1, *sample_wosty.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 

        re_motion = data.dataset.t2m_dataset.inv_transform(re_motion.cpu().detach().numpy().transpose(0, 2, 3, 1))
        re_motion = torch.from_numpy(re_motion).to(args.device)
        re_motion = recover_from_ric(re_motion, n_joints)
        re_motion = re_motion.view(-1, *re_motion.shape[2:]).permute(0, 2, 3, 1)

    
    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    sample_wosty = model.rot2xyz(x=sample_wosty, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                            jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                            get_rotations_back=False) 
    re_motion = model.rot2xyz(x=re_motion, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                            jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                            get_rotations_back=False) 

    caption = model_rawkwargs['y']['text']

    sample_wosty = sample_wosty.cpu().numpy().transpose(0, 3, 1, 2)
    re_motion = re_motion.cpu().numpy().transpose(0, 3, 1, 2)
    sample_array = plot_3d_array([sample_wosty[0], None, paramUtil.t2m_kinematic_chain, caption[0]])
    imageio.mimsave(os.path.join(out_path, 'sample_results.gif'), np.array(sample_array), fps=20)
    re_array = plot_3d_array([re_motion[0], None, paramUtil.t2m_kinematic_chain, caption[0]+"_re"])
    imageio.mimsave(os.path.join(out_path, 'sample_results_rec.gif'), np.array(re_array), fps=20)
    
    


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    
    sty_data = get_dataset_loader(name=args.style_dataset,
                                  batch_size=args.batch_size,
                                  num_frames=max_frames, 
                                  split='test',
                                  hml_mode="text_only")
    
    return data, sty_data


if __name__ == "__main__":
    main()
