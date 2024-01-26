import numpy as np
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil
import torch
from data_loaders.humanml.common.skeleton import Skeleton
import os
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.common.bvh_utils import read_bvh
from data_loaders.humanml.common.rotation import wrap, quat_fk


# source_dir = '/data/hulei/Projects/Style100_2_HumanML/style_xia_with_rotation/new_bvh'
# our_bvh_dir = './eval_results/stylexia_correct_Ls/'
# motionpuzzle_dir = '/data/hulei/OpenProjects/MotionPuzzle/eval_resuls_stylexia/'
# finestyle_dir = '/mnt/data_0/hulei/StyleProjects/Fine-Style/eval_results/stylexia/'
# umst_dir = '/mnt/data_0/hulei/StyleProjects/Style_transfer/deep-motion-editing/style_transfer/eval_results/stylexia'
# root_dir =  './userstudy/stylexia/'

# methods_dir = [our_bvh_dir, motionpuzzle_dir, finestyle_dir, umst_dir]
# methods_names = ['ours', 'motion_puzzle', 'finestyle', 'umst']

# kinematic_chain = paramUtil.xia_kinematic_chain
# real_offset = paramUtil.xia_real_offsets
# ee_names = ["rtoes", 'ltoes', 'lfoot', 'rfoot']
# skeleton = Skeleton(torch.Tensor(paramUtil.xia_raw_offsets), kinematic_chain, torch.device('cuda:0'))

# result_list = [file for file in os.listdir(our_bvh_dir) if file.endswith('.bvh') and file[-8:-4]!='woik']
# np.random.shuffle(result_list)

# gt_frames_per_sample = {}

# for i, file in enumerate(result_list):
#     if i > 19:
#         break
#     if not os.path.exists(os.path.join(root_dir, file[:-4])):
#         os.mkdir(os.path.join(root_dir, file[:-4]))
#     new_root = os.path.join(root_dir, file[:-4])
#     c_index = file.index('Content_')
#     content_file_path = file[c_index+8:]
#     c_anim = read_bvh(os.path.join(source_dir, content_file_path))
    
#     _, gp_content = wrap(quat_fk, c_anim.quats, c_anim.pos, c_anim.parents)
#     content_caption = content_file_path.split('_')
#     content_label = content_caption[1].split(" ")
#     if len(content_label) > 1:
#         content_label = content_label[1]
#     else:
#         content_label = content_label[0]
#     content_caption = " ".join([content_caption[0][3:], content_label[:-4]])
#     content_save_path = os.path.join(new_root, content_file_path.replace('.bvh', '.mp4'))
#     try:
#         plot_3d_motion(content_save_path, kinematic_chain, gp_content, title=content_caption,
#                         dataset='stylexia_posrot', fps=10, vis_mode='gt',
#                         gt_frames=gt_frames_per_sample.get(0, []), painting_features=['root_horizontal'])
#     except:
#         pass

#     # style file path
#     style_file_path = file[6:c_index-1] + '.bvh'
#     s_anim = read_bvh(os.path.join(source_dir, style_file_path))
    
#     _, gp_style = wrap(quat_fk, s_anim.quats, s_anim.pos, s_anim.parents)
#     style_caption = style_file_path.split('_')
#     style_label = style_caption[1].split(" ")
#     if len(style_label) > 1:
#         style_label = style_label[1]
#     else:
#         style_label = style_label[0]
#     style_caption = " ".join([style_caption[0][3:], style_label[:-4]])
#     style_save_path = os.path.join(new_root, style_file_path.replace('.bvh', '.mp4'))
#     try:
#         plot_3d_motion(style_save_path, kinematic_chain, gp_style, title=style_caption,
#                         dataset='stylexia_posrot', fps=10, vis_mode='gt',
#                         gt_frames=gt_frames_per_sample.get(0, []), painting_features=['root_horizontal'])
#     except:
#         pass

#     for i, name in enumerate(methods_names):
#         method_dir = methods_dir[i]
#         generated_anim = read_bvh(os.path.join(method_dir, file))
#         _, gp_generated = wrap(quat_fk, generated_anim.quats, generated_anim.pos, generated_anim.parents)
#         results_save_path = os.path.join(new_root, name + "_" + file.replace('.bvh', '.mp4'))
#         try:
#             plot_3d_motion(results_save_path, kinematic_chain, gp_generated, title=" ",
#                     dataset='stylexia_posrot', fps=10, vis_mode='root_horizontal',
#                     gt_frames=gt_frames_per_sample.get(0, []), painting_features=['root_horizontal'])
#         except:
#             pass
        


source_dir = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2_with_rotation/new_bvh'
our_bvh_dir = './eval_results/bandai_correct_Ls/'
motionpuzzle_dir = '/data/hulei/OpenProjects/MotionPuzzle/eval_resuls_bandai/'
finestyle_dir = '/mnt/data_0/hulei/StyleProjects/Fine-Style/eval_results/bandai/'
umst_dir = '/mnt/data_0/hulei/StyleProjects/Style_transfer/deep-motion-editing/style_transfer/eval_results/bandai'
root_dir =  './userstudy/bandai/'

methods_dir = [our_bvh_dir, motionpuzzle_dir, finestyle_dir, umst_dir]
methods_names = ['ours', 'motion_puzzle', 'finestyle', 'umst']

kinematic_chain = paramUtil.bandai_kinematic_chain
real_offset = paramUtil.bandai_real_offsets


ee_names = ["Toes_R", 'Toes_L', 'Foot_L', 'Foot_R']
skeleton = Skeleton(torch.Tensor(paramUtil.bandai_raw_offsets), kinematic_chain, torch.device('cuda:0'))

result_list = [file for file in os.listdir(our_bvh_dir) if file.endswith('.bvh') and file[-8:-4]!='woik']
np.random.shuffle(result_list)

gt_frames_per_sample = {}


all_err = []
for i, file in enumerate(result_list):
    if i > 400:
        break
    if not os.path.exists(os.path.join(root_dir, file[:-4])):
         os.mkdir(os.path.join(root_dir, file[:-4]))
    new_root = os.path.join(root_dir, file[:-4])
    c_index = file.index('Content_')
    content_file_path = file[c_index+8:]
    c_anim = read_bvh(os.path.join(source_dir, content_file_path))
    
    _, gp_content = wrap(quat_fk, c_anim.quats, c_anim.pos, c_anim.parents)
    content_caption = content_file_path.split('_')
    content_label = content_caption[1]
    content_caption = " ".join([content_label, content_caption[-2]])
    content_save_path = os.path.join(new_root, content_file_path.replace('.bvh', '.mp4'))
    # try:
    #     plot_3d_motion(content_save_path, kinematic_chain, gp_content, title=content_caption,
    #                     dataset='bandai-2_posrot', fps=10, vis_mode='gt',
    #                     gt_frames=gt_frames_per_sample.get(0, []), painting_features=['root_horizontal'])
    # except:
    #     pass
    
    
    # style file path
    style_file_path = file[6:c_index-1] + '.bvh'
    s_anim = read_bvh(os.path.join(source_dir, style_file_path))
    
    _, gp_style = wrap(quat_fk, s_anim.quats, s_anim.pos, s_anim.parents)
    style_caption = style_file_path.split('_')
    style_label = style_caption[1]
    style_caption = " ".join([style_label, style_caption[-2]])
    style_save_path = os.path.join(new_root, style_file_path.replace('.bvh', '.mp4'))
    # try:
    #     plot_3d_motion(style_save_path, kinematic_chain, gp_style, title=style_caption,
    #                     dataset='bandai-2_posrot', fps=10, vis_mode='gt',
    #                     gt_frames=gt_frames_per_sample.get(0, []), painting_features=['root_horizontal'])
        
    # except:
    #     pass 

    all_results = []
    for i, name in enumerate(methods_names):
        method_dir = methods_dir[i]
        generated_anim = read_bvh(os.path.join(method_dir, file))
        _, gp_generated = wrap(quat_fk, generated_anim.quats, generated_anim.pos, generated_anim.parents)
        all_results.append(gp_generated)
        results_save_path = os.path.join(new_root, name + "_" + file.replace('.bvh', '.mp4'))
        # try:
        #     plot_3d_motion(results_save_path, kinematic_chain, gp_generated, title=" ",
        #             dataset='bandai-2_posrot', fps=10, vis_mode='root_horizontal',
        #             gt_frames=gt_frames_per_sample.get(0, []), painting_features=['root_horizontal'])
        # except:
        #     pass

    min_len = min([res.shape[0] for res in all_results])
    all_results = [res[:min_len, ...] for res in all_results]
    min_err = 99999
    for i, result in enumerate(all_results):
        if i == 0:
            continue
        err = np.mean(np.linalg.norm(all_results[0] - all_results[i], ord=2, axis=-1))
        if err < min_err:
            min_err = err
    all_err.append([file, min_err])

sorted_data = sorted(all_err, key=lambda x: x[1], reverse=True)
# sorted_files = zip(*sorted_data)[0]

for i, item in enumerate(zip(*sorted_data)):
    if i == 0:
        sorted_files = item
        
with open(os.path.join(root_dir, 'chose_list.txt'), 'w') as f:
    for bvh in sorted_files:
        f.write(bvh + '\n')

    
         
    


        
    
    


    