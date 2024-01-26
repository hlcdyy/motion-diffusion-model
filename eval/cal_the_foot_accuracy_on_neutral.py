import os
import numpy as np
from data_loaders.humanml.common.bvh_utils import read_bvh, get_foot_contact_by_vel_acc, get_ee_id_by_names, get_foot_contact_by_vel3
from data_loaders.humanml.common.rotation import wrap, quat_fk
from scipy import io

source_dir = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2_with_rotation/new_bvh'
# bvh_dir = './eval_neutral/bandai_noise1000_wotrajectory/'
bvh_dir = './eval_results/bandai_correct_Ls'
ref_dir = './eval_ablation/bandai_K400'
# files = [os.path.join(bvh_dir, file) for file in os.listdir(bvh_dir) if file.endswith('.bvh')]

files = [os.path.join(bvh_dir, file) for file in os.listdir(ref_dir) if file.endswith('.bvh')]

c_list = ["run", "walk", "walk-turn-right", 'walk-turn-left']

count = 0
acc = 0
for file in files:
    # source_file = file.replace('Neutral_', "").split("/")[-1]
    source_file = file[file.index('Content_'):]
    source_file = source_file.replace('Content_', "")

    content = source_file.split("_")[1]
    if content not in c_list or file.split("/")[-1][-8:-4] == 'woik':
        continue
    source_file = os.path.join(source_dir, source_file)
    source_anim = read_bvh(source_file)
    _, source_motion = wrap(quat_fk, source_anim.quats, source_anim.pos, source_anim.parents)
    anim = read_bvh(file)
    _, neutral_motion = wrap(quat_fk, anim.quats, anim.pos, anim.parents)

    if source_motion.shape[0] > neutral_motion.shape[0]:
        source_motion = source_motion[:neutral_motion.shape[0]]
    

    ee_ids = get_ee_id_by_names(source_anim.bones, ["Toes_R", 'Toes_L', 'Foot_L', 'Foot_R'])
    # tmp = source_motion[1:, ee_ids, :] -  source_motion[:-1, ee_ids, :]
    # tmp = np.linalg.norm(tmp, ord=2, axis=-1)
    # tmp1 = neutral_motion[1:, ee_ids, :] -  neutral_motion[:-1, ee_ids, :]
    # tmp1 = np.linalg.norm(tmp1, ord=2, axis=-1)
    # io.savemat('tmp_vel.mat', {"vel": tmp, "neutral_vel": tmp1})
    # exit()
    contact_source, _, _ = get_foot_contact_by_vel3(source_motion, ee_ids, ref_height=None, thr=0.02, use_butterworth=False)
    contact_neutral, _, _ = get_foot_contact_by_vel3(neutral_motion, ee_ids, ref_height=None, thr=0.02, use_butterworth=False)
    
    hit = contact_neutral * contact_source
    
    accuracy = np.sum(hit)/np.sum(contact_source)
    acc = (accuracy * np.sum(contact_source) + acc * count)/(count+ np.sum(contact_source))
    count += np.sum(contact_source)

print(acc)
    