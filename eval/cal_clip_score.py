import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
from utils.fixseed import fixseed
from utils.parser_util import eval_inpainting_style_args
from utils import dist_util
from data_loaders.humanml.common.bvh_utils import process_file_with_rotation, read_bvh
from data_loaders.humanml.common.rotation import wrap, quat_fk
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_serval_diffusion, creat_ddpm_ddim_diffusion
import torch 
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.tensors import collate
from utils.model_util import  load_model_wo_controlmdm, creat_serval_diffusion,load_model_wo_moenc

from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from visualize.vis_utils import joints2bvh
import data_loaders.humanml.utils.paramUtil as paramUtil
from numpy.lib.format import open_memmap
import pickle
import numpy as np

args = eval_inpainting_style_args()
max_frames = 196 if args.dataset in ['kit', 'humanml',"bandai-1_posrot", "bandai-2_posrot"] else 60
max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames
fps = 12.5 if args.dataset == 'kit' else 20
dist_util.setup_dist(args.device)

print("creating model and diffusion...")
DiffusionClass = InpaintingGaussianDiffusion
if args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
    from model.mdm_forstyledataset import StyleDiffusion

cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

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

dataloder = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='test')

# real_motion_dir = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2_with_rotation/new_bvh'
# eval_dir = './eval_neutral/bandai_noise1000'
# real_file_name = os.listdir(real_motion_dir)
# real_list = []
# for file in real_file_name:
#     real_list.append([file, file.split('_')[1], file.split('_')[2]])
# real_features = {}
# all_clip_scores = []
# all_sentence_scores = []
# all_style_word_scores = []
# files = [os.path.join(eval_dir, file) for file in os.listdir(eval_dir) if file.endswith('.bvh') and file[-8:-4] != 'woik']
# bandai_raw_offsets, bandai_chain = paramUtil.bandai_raw_offsets, paramUtil.bandai_kinematic_chain
# bandai_raw_offsets = torch.Tensor(bandai_raw_offsets)
# for file in files:
#     content = file.split("/")[-1].split("_")[2]
#     style = 'normal'
#     anim = read_bvh(file)
#     _, joints = wrap(quat_fk, anim.quats, anim.pos, anim.parents)
#     joints = joints.astype(np.float32)
#     data, _, _, _ = process_file_with_rotation(joints, anim.quats, face_joint_indx=[17, 13, 9, 5], fid_l=[15, 16], fid_r=[19, 20], feet_thre=0.002, 
#                                            n_raw_offsets=bandai_raw_offsets, kinematic_chain=bandai_chain)
    
#     motion, m_length = dataloder.dataset.t2m_dataset.process_np_motion(data)
#     collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': data.shape[0]}] * 1
#     # caption = 'None'
#     content_text = content.split('-')
#     content_text[0] = content_text[0] + 's'
#     content_text = " ".join(content_text)
#     # texts = ['A person '+ content_text+ " normal"]
#     texts = ["normal"] 
#     collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
#     _, model_kwargs = collate(collate_args) 
#     model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}    


#     # motion = motion.unsqueeze(0).unsqueeze(2)
#     motion = motion[None, :, None, :]
#     motion = motion.transpose(0, 3, 2, 1)
#     motion = torch.Tensor(motion).float().to(dist_util.dev())
#     with torch.no_grad():
#         mu, text_features = model.motion_enc(motion, model_kwargs['y'])
    
#     gt_feat = real_features.setdefault(content+'_'+style, [])
#     if len(gt_feat) ==0 :
#         for item in real_list:
#             if item[1] == content and item[2] == style:
#                 gt_file = os.path.join(real_motion_dir, item[0])
#                 gt_anim = read_bvh(gt_file)
#                 _, gt_joints = wrap(quat_fk, gt_anim.quats, gt_anim.pos, gt_anim.parents)
#                 gt_joints = gt_joints.astype(np.float32)
#                 gt_data, _, _, _ = process_file_with_rotation(gt_joints, gt_anim.quats, face_joint_indx=[17, 13, 9, 5], fid_l=[15, 16], fid_r=[19, 20], feet_thre=0.002, n_raw_offsets=bandai_raw_offsets, kinematic_chain=bandai_chain)
    
#                 motion, m_length = dataloder.dataset.t2m_dataset.process_np_motion(gt_data)
#                 collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': data.shape[0]}] * 1
#                 _, model_kwargs = collate(collate_args) 
#                 model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}    
#                 motion = motion[None, :, None, :]
#                 motion = motion.transpose(0, 3, 2, 1)
#                 motion = torch.Tensor(motion).float().to(dist_util.dev())
#                 with torch.no_grad():
#                     gt_mu, _ = model.motion_enc(motion, model_kwargs['y'])
#                 real_features[content+'_'+style].append(gt_mu)
#         real_features[content+'_'+style] = torch.cat(real_features[content+'_'+style], 0)
#     gt_feat = real_features[content+'_'+style]
    
#     features_norm = mu / mu.norm(dim=-1, keepdim=True)
#     gt_features_norm = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
#     text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)
#     clip_score = cosine_sim(features_norm, gt_features_norm).mean()
#     all_clip_scores.append(clip_score.detach().cpu().numpy())
#     all_sentence_scores.append(cosine_sim(features_norm, text_features_norm).detach().cpu().numpy())

# assert len(all_clip_scores) == 360
# print(np.mean(all_clip_scores))
# print(np.mean(all_sentence_scores))

real_motion_dir = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2_with_rotation/new_bvh'
eval_dir = './eval_ablation/bandai_K100'
# data_dir = './eval_results/bandai_correct_Ls/'
real_file_name = os.listdir(real_motion_dir)
real_list = []
for file in real_file_name:
    real_list.append([file, file.split('_')[1], file.split('_')[2]])
real_features = {}
all_clip_scores = []
all_sentence_scores = []
all_style_word_scores = []
files = [os.path.join(eval_dir, file) for file in os.listdir(eval_dir) if file.endswith('.bvh') and file[-8:-4] != 'woik']
# files = [os.path.join(data_dir, file) for file in os.listdir(eval_dir) if file.endswith('.bvh') and file[-8:-4] != 'woik']

bandai_raw_offsets, bandai_chain = paramUtil.bandai_raw_offsets, paramUtil.bandai_kinematic_chain
bandai_raw_offsets = torch.Tensor(bandai_raw_offsets)
for file in files:
    content = file.split("/")[-1].split("_")[-3]
    style = file.split("/")[-1].split("_")[3]
    anim = read_bvh(file)
    _, joints = wrap(quat_fk, anim.quats, anim.pos, anim.parents)
    joints = joints.astype(np.float32)
    data, _, _, _ = process_file_with_rotation(joints, anim.quats, face_joint_indx=[17, 13, 9, 5], fid_l=[15, 16], fid_r=[19, 20], feet_thre=0.002, 
                                              n_raw_offsets=bandai_raw_offsets, kinematic_chain=bandai_chain)
    
    motion, m_length = dataloder.dataset.t2m_dataset.process_np_motion(data)
    collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': data.shape[0]}] * 1
    # caption = 'None'
    content_text = content.split('-')
    content_text[0] = content_text[0] + 's'
    content_text = " ".join(content_text)
    # texts = ['A person '+ content_text+ " normal"]
    # texts = ["normal"] 
    # texts = ['A person'+ content_text + style]
    texts = [style]
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args) 
    model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}    


    # motion = motion.unsqueeze(0).unsqueeze(2)
    motion = motion[None, :, None, :]
    motion = motion.transpose(0, 3, 2, 1)
    motion = torch.Tensor(motion).float().to(dist_util.dev())
    with torch.no_grad():
        mu, text_features = model.motion_enc(motion, model_kwargs['y'])
    
    gt_feat = real_features.setdefault(content+'_'+style, [])
    if len(gt_feat) ==0 :
        for item in real_list:
            if item[1] == content and item[2] == style:
                gt_file = os.path.join(real_motion_dir, item[0])
                gt_anim = read_bvh(gt_file)
                _, gt_joints = wrap(quat_fk, gt_anim.quats, gt_anim.pos, gt_anim.parents)
                gt_joints = gt_joints.astype(np.float32)
                gt_data, _, _, _ = process_file_with_rotation(gt_joints, gt_anim.quats, face_joint_indx=[17, 13, 9, 5], fid_l=[15, 16], fid_r=[19, 20], feet_thre=0.002, n_raw_offsets=bandai_raw_offsets, kinematic_chain=bandai_chain)
    
                motion, m_length = dataloder.dataset.t2m_dataset.process_np_motion(gt_data)
                collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': data.shape[0]}] * 1
                _, model_kwargs = collate(collate_args) 
                model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}    
                motion = motion[None, :, None, :]
                motion = motion.transpose(0, 3, 2, 1)
                motion = torch.Tensor(motion).float().to(dist_util.dev())
                with torch.no_grad():
                    gt_mu, _ = model.motion_enc(motion, model_kwargs['y'])
                real_features[content+'_'+style].append(gt_mu)
        real_features[content+'_'+style] = torch.cat(real_features[content+'_'+style], 0)
    gt_feat = real_features[content+'_'+style]
    
    features_norm = mu / mu.norm(dim=-1, keepdim=True)
    gt_features_norm = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
    text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)
    clip_score = cosine_sim(features_norm, gt_features_norm).mean()
    all_clip_scores.append(clip_score.detach().cpu().numpy())
    all_sentence_scores.append(cosine_sim(features_norm, text_features_norm).detach().cpu().numpy())

# assert len(all_clip_scores) == 360
print(len(all_clip_scores))
print(np.mean(all_clip_scores))
print(np.mean(all_sentence_scores))

