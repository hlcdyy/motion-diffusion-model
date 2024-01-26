from numpy.lib.format import open_memmap
import pickle
import numpy as np
from data_loaders.humanml.common.bvh_utils import read_bvh
from data_loaders.humanml.common.rotation import quat_fk, wrap
import os
from tqdm import tqdm

# bvh_dir = './eval_results/bandai_correct_Ls/'
# bvh_dir = '/mnt/data_0/hulei/StyleProjects/Fine-Style/eval_results/bandai/'
# bvh_dir = '/mnt/data_0/hulei/StyleProjects/Style_transfer/deep-motion-editing/style_transfer/eval_results/bandai/'
bvh_dir = './eval_ablation/bandai_K300'
# files = [os.path.join(bvh_dir, file) for file in os.listdir(bvh_dir) if file.endswith('.bvh')]
ref_dir = './eval_ablation/bandai_K600'
motion_dir = './eval_results/bandai_correct_Ls/'
files = [os.path.join(motion_dir, file) for file in os.listdir(ref_dir) if file.endswith('.bvh')]


all_trans_motion = []
all_trans_motion_woik = []
all_style_labels = []
all_content_labels = []
all_names = []
all_length = []
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


for file in tqdm(files):
    # if file[-8:-4] == 'woik':
    if file[-7:-4] != 'fix':
        anim_woik = read_bvh(file)
    else:
        anim = read_bvh(file)
    
    # if file[-8:-4] != 'woik':
    if file[-7:-4] != 'fix':
        content_index = file.split('/')[-1].index('Content_')
        content_file = file.split('/')[-1][content_index:-4]
        contetn_name = content_file.split("_")[2]
        style_file = file.split('/')[-1][:content_index-1]
        style_name = style_file.split("_")[-2]

        all_style_labels.append(style_id_mapping[style_name])
        all_content_labels.append(content_id_mapping[contetn_name])
        all_names.append(file.split('/')[-1][:-4])
    
    # if file[-8:-4] != 'woik':
    if file[-7:-4] == 'fix':
        gr, gp = wrap(quat_fk, anim.quats, anim.pos, anim.parents)
        all_trans_motion.append(gp)

    else:
        gr, gp = wrap(quat_fk, anim_woik.quats, anim_woik.pos, anim_woik.parents)
        all_trans_motion_woik.append(gp)
        all_length.append(gp.shape[0])


max_frame_eval = 350
joint_num = 21

all_length = np.array(all_length, dtype=int)    
np.save('{}/{}_length.npy'.format(bvh_dir, "output"), all_length)

# fp = open_memmap(
#     '{}/{}_data.npy'.format(bvh_dir, "output"),
#     dtype='float32',
#     mode='w+',
#     shape=(len(all_trans_motion), 3, max_frame_eval, joint_num, 1))

fp_woik = open_memmap(
    '{}/{}_woik_data.npy'.format(bvh_dir, "output"),
    dtype='float32',
    mode='w+',
    shape=(len(all_trans_motion_woik), 3, max_frame_eval, joint_num, 1))

for i, s in enumerate(all_names):
    # data = all_trans_motion[i].transpose(2, 0, 1)
    # fp[i, :, 0:data.shape[1], :, 0] = data
    
    data_woik = all_trans_motion_woik[i].transpose(2, 0, 1)
    fp_woik[i, :, 0:data_woik.shape[1], :, 0] = data_woik


with open('{}/{}_label.pkl'.format(bvh_dir, "content"), 'wb') as f:
    pickle.dump((all_names, all_content_labels), f)

with open('{}/{}_label.pkl'.format(bvh_dir, "style"), 'wb') as f:
    pickle.dump((all_names, all_style_labels), f)


# bvh_dir = './eval_results/stylexia_ae/'
# files = [os.path.join(bvh_dir, file) for file in os.listdir(bvh_dir) if file.endswith('.bvh')]
# all_trans_motion = []
# all_trans_motion_woik = []
# all_style_labels = []
# all_content_labels = []
# all_names = []
# all_length = []
# content_id_mapping = {
#             "walking": 0,
#             "running": 1,
#             "jumping":2,
#             "punching":3,
#             "kicking":4,
#             "transitions":5
#         }
# style_id_mapping = {
#             "angry": 0,
#             "childlike":1,
#             "depressed":2,
#             "neutral":3,
#             "old":4,
#             "proud":5,
#             "sexy":6,
#             "strutting":7
#         }

# id_style_mapping = ["angry",
#     "childlike",
#     "depressed",
#     "neutral",
#     "old",
#     "proud",
#     "sexy",
#     "strutting"]


# for file in files:
#     if file[-8:-4] == 'woik':
#         anim_woik = read_bvh(file)
#     else:
#         anim = read_bvh(file)

#     if file[-8:-4] != 'woik':
#         content_index = file.split('/')[-1].index('Content_')
#         content_file = file.split('/')[-1][content_index:-4]
        
#         content_name = content_file.split("_")[2]
#         c_content_name = content_name.split(" ")
#         if len(c_content_name) > 1:
#             c_content = c_content_name[1]
#         else:
#             c_content = c_content_name[0]    
            
    
#         style_file = file.split('/')[-1][:content_index-1]
        
#         style_name = style_file.split("_")[1][3:]
#         all_style_labels.append(style_id_mapping[style_name])
        
#         all_content_labels.append(content_id_mapping[c_content])
#         all_names.append(file.split('/')[-1][:-4])
        
    
#     if file[-8:-4] == 'woik':
#         gr, gp = wrap(quat_fk, anim_woik.quats, anim_woik.pos, anim_woik.parents)
#         all_trans_motion_woik.append(gp)
#     else:
#         gr, gp = wrap(quat_fk, anim.quats, anim.pos, anim.parents)
#         all_trans_motion.append(gp)
#         all_length.append(gp.shape[0])



# max_frame_eval = 76
# joint_num = 20

# all_length = np.array(all_length, dtype=int)    
# np.save('{}/{}_length.npy'.format(bvh_dir, "output"), all_length)

# fp = open_memmap(
#     '{}/{}_data.npy'.format(bvh_dir, "output"),
#     dtype='float32',
#     mode='w+',
#     shape=(len(all_trans_motion), 3, max_frame_eval, joint_num, 1))

# fp_woik = open_memmap(
#     '{}/{}_woik_data.npy'.format(bvh_dir, "output"),
#     dtype='float32',
#     mode='w+',
#     shape=(len(all_trans_motion_woik), 3, max_frame_eval, joint_num, 1))

# for i, s in enumerate(all_names):
#     data = all_trans_motion[i].transpose(2, 0, 1)
#     fp[i, :, 0:data.shape[1], :, 0] = data
    
#     data_woik = all_trans_motion_woik[i].transpose(2, 0, 1)
#     fp_woik[i, :, 0:data_woik.shape[1], :, 0] = data_woik


# with open('{}/{}_label.pkl'.format(bvh_dir, "content"), 'wb') as f:
#     pickle.dump((all_names, all_content_labels), f)

# with open('{}/{}_label.pkl'.format(bvh_dir, "style"), 'wb') as f:
#     pickle.dump((all_names, all_style_labels), f)


# bvh_dir = './eval_neutral/bandai_noise1000/'
# files = [os.path.join(bvh_dir, file) for file in os.listdir(bvh_dir) if file.endswith('.bvh')]
# all_trans_motion = []
# all_trans_motion_woik = []
# all_style_labels = []
# all_content_labels = []
# all_names = []
# all_length = []
# content_id_mapping = {
#             "walk":0,
#             "walk-turn-left":1,
#             "walk-turn-right":2,
#             "run":3,
#             "wave-both-hands":4,
#             "wave-left-hand":5,
#             "wave-right-hand":6,
#             "raise-up-both-hands":7,
#             "raise-up-left-hand":8,
#             "raise-up-right-hand":9,
#             }

# style_id_mapping = {
#     "active":0,
#     "elderly":1,
#     "exhausted":2,
#     "feminine":3,
#     "masculine":4,
#     "normal":5,
#     "youthful":6
# }
# id_style_mapping = ["active",
#     "elderly",
#     "exhausted",
#     "feminine",
#     "masculine",
#     "normal",
#     "youthful"]


# for file in tqdm(files):
#     if file[-8:-4] == 'woik':
#         anim_woik = read_bvh(file)
#     else:
#         anim = read_bvh(file)
    
#     if file[-8:-4] != 'woik':
       
#         content_file = file.split('/')[-1][8:-4]
#         contetn_name = content_file.split("_")[1]
#         style_name = 'normal'
#         all_style_labels.append(style_id_mapping[style_name])
#         all_content_labels.append(content_id_mapping[contetn_name])
#         all_names.append(file.split('/')[-1][:-4])
    
#     if file[-8:-4] == 'woik':
#         gr, gp = wrap(quat_fk, anim_woik.quats, anim_woik.pos, anim_woik.parents)
#         all_trans_motion_woik.append(gp)
#     else:
#         gr, gp = wrap(quat_fk, anim.quats, anim.pos, anim.parents)
#         all_trans_motion.append(gp)
#         all_length.append(gp.shape[0])


# max_frame_eval = 350
# joint_num = 21

# all_length = np.array(all_length, dtype=int)    
# np.save('{}/{}_length.npy'.format(bvh_dir, "output"), all_length)

# fp = open_memmap(
#     '{}/{}_data.npy'.format(bvh_dir, "output"),
#     dtype='float32',
#     mode='w+',
#     shape=(len(all_trans_motion), 3, max_frame_eval, joint_num, 1))

# fp_woik = open_memmap(
#     '{}/{}_woik_data.npy'.format(bvh_dir, "output"),
#     dtype='float32',
#     mode='w+',
#     shape=(len(all_trans_motion_woik), 3, max_frame_eval, joint_num, 1))

# for i, s in enumerate(all_names):
#     data = all_trans_motion[i].transpose(2, 0, 1)
#     fp[i, :, 0:data.shape[1], :, 0] = data
    
#     data_woik = all_trans_motion_woik[i].transpose(2, 0, 1)
#     fp_woik[i, :, 0:data_woik.shape[1], :, 0] = data_woik


# with open('{}/{}_label.pkl'.format(bvh_dir, "content"), 'wb') as f:
#     pickle.dump((all_names, all_content_labels), f)

# with open('{}/{}_label.pkl'.format(bvh_dir, "style"), 'wb') as f:
#     pickle.dump((all_names, all_style_labels), f)
