# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import predict_trajectory_args 
from utils.model_util import creat_motion_trajectory, load_model_wo_clip
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_from_vel
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_array
import imageio
import shutil
from data_loaders.tensors import collate
from copy import deepcopy
from tqdm import tqdm


def main():
    args = predict_trajectory_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = max_frames
    is_using_data = False
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'reconstruction_{}_{}_seed{}'.format(name, niter, args.seed))
        # if args.text_prompt != '':
        #     out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        # elif args.input_text != '':
        #     out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model = creat_motion_trajectory(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    
    model_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, model_dict)

    # if args.guidance_param != 1:
    #     mdm_model = ClassifierFreeSampleModel(mdm_model)   # wrapping model with the classifier-free sampler

    model.to(dist_util.dev())
    model.eval() # disable random masking
    
    if is_using_data:
        iterator = iter(data)
        t2m_motion, model_kwargs = next(iterator)
        t2m_motion = t2m_motion.to(dist_util.dev())
        model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}
    else:
        # motion_dir = './videos/res_xixiyu/'
        # path = os.path.join(motion_dir, 'res.pk')
        # fps_video = 30
        # start_second = 52.4 
        # end_second = 55.5
        # from utils.process_smpl_from_hybrik import amass_to_pose, pos2hmlrep
        # pose_seq_np_n, joints = amass_to_pose(path, fps_video, trans_path="", with_trans=False)
        # path = pos2hmlrep(joints)
        # path = path[int(20*start_second):int(20*end_second)]

        motion_dir = './videos/H36M_S6/glamr_static_Walking_1.60457274/pose_est'
        path = os.path.join(motion_dir, 'pose.pkl')
        fps_video = 30
        start_second = 17
        end_second = 22
        from utils.process_smpl_from_hybrik import amass_to_pose, pos2hmlrep
        pose_seq_np_n, joints = amass_to_pose(path, fps_video, trans_path="", with_trans=False)
        path = pos2hmlrep(joints)
        path = path[int(20*start_second):int(20*end_second)] # after process fixed 20 fps

        # motion_dir = '/data/hulei/OpenProjects/HumanML3D/HumanML3D/new_joint_vecs'
        # path = os.path.join(motion_dir, '003027.npy')
        t2m_motion, m_length = data.dataset.t2m_dataset.process_np_motion(path)
        t2m_motion = torch.Tensor(t2m_motion.T).unsqueeze(1).unsqueeze(0)
        t2m_motion = t2m_motion.to(dist_util.dev()) 

        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1
   
        texts = ["input_motion"] * args.num_samples

        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)
        model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}
        
    all_motions = []
    all_lengths = []
    all_text = []

    # add CFG scale to batch    
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param   
    
    with torch.no_grad():
        pred_root = model(t2m_motion, **model_kwargs)
    
    # tmp = pred_root.squeeze().permute(1, 0)
    pred_motion = torch.cat((pred_root, t2m_motion[:, 4:, ...]), 1)
    
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    if model.data_rep == 'hml_vec':
        n_joints = 22 if pred_motion.shape[1] == 263 else 21
        
        sample_predict = data.dataset.t2m_dataset.inv_transform(pred_motion.detach().cpu().permute(0, 2, 3, 1)).float()
        sample_predict = recover_from_ric(sample_predict, n_joints) # B 1 T J 3 
        sample_predict = sample_predict.view(-1, *sample_predict.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 

        sample_gt = data.dataset.t2m_dataset.inv_transform(t2m_motion.cpu().permute(0, 2, 3, 1)).float()
        sample_gt = recover_from_vel(sample_gt, n_joints)
        sample_gt = sample_gt.view(-1, *sample_gt.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 

    
    caption = model_kwargs['y']['text']
    
    sample_predict = sample_predict.cpu().numpy().transpose(0, 3, 1, 2)
    sample_gt = sample_gt.cpu().numpy().transpose(0, 3, 1, 2)
  
    sample_array = plot_3d_array([sample_predict[0][:model_kwargs["y"]["lengths"][0]], None, paramUtil.t2m_kinematic_chain, caption[0] + "_predict_trajectory"])
    imageio.mimsave(os.path.join(out_path, 'pred_root_trajectory_results.gif'), np.array(sample_array), duration=int(model_kwargs["y"]["lengths"][0]/20))
    gt_array = plot_3d_array([sample_gt[0][:model_kwargs["y"]["lengths"][0]], None, paramUtil.t2m_kinematic_chain, caption[0]])
    imageio.mimsave(os.path.join(out_path, 'gt_results.gif'), np.array(gt_array), duration=int(model_kwargs["y"]["lengths"][0]/20))
    

def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='eval')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames

    return data

if __name__ == "__main__":
    main()
