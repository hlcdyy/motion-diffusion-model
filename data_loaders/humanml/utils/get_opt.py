import os
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    
    # using under options
    if opt.dataset_name == 't2m':
        opt.data_root = '/data/hulei/OpenProjects/HumanML3D/HumanML3D' # use our revised representation
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'  
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
    elif opt.dataset_name == 'style100':
        opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/style100_data/'
        opt.t2m_root = '/data/hulei/OpenProjects/HumanML3D/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_description = './data_loaders/style100/Dataset_List.csv'
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196

    elif opt.dataset_name in ['bandai-1', 'bandai-2']:
        if opt.dataset_name == 'bandai-1':
            opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-1/'
        else:
            opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2/' # for the old tpose save_bandai and save_bandai_train
        opt.t2m_root = '/data/hulei/OpenProjects/HumanML3D/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        
    elif opt.dataset_name in ['bandai-1_posrot', 'bandai-2_posrot']:
        if opt.dataset_name == 'bandai-1_posrot':
            opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-1_with_rotation/'
        else:
            # opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/Bandai-Namco-Research-Motiondataset-2_with_rotation/'  # for the old tpose save_bandai and save_bandai_train
            opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/bandai-2_with_rotation_normaltpose/'  # for new model save_bandai_train_newtpose
        opt.t2m_root = '/data/hulei/Projects/Style100_2_HumanML/bandai-2_with_rotation_normaltpose/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        opt.dim_pose = 190
        opt.max_motion_length = 196

    elif opt.dataset_name == 'stylexia_posrot':
        
        opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/style_xia_with_rotation/'
        opt.t2m_root = '/data/hulei/Projects/Style100_2_HumanML/style_xia_with_rotation/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 20
        opt.dim_pose = 181
        opt.max_motion_length = 76

    elif opt.dataset_name == 'AIST_posrot':
        opt.data_root = '/data/hulei/Projects/Style100_2_HumanML/AIST++_with_rotation/'
        opt.t2m_root = '/data/hulei/Projects/Style100_2_HumanML/AIST++_with_rotation/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.dim_pose = 199
        opt.max_motion_length = 196
    
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt