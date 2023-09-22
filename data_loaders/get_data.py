from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_style_collate, style_pairs_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name in ['bandai-1', 'bandai-2', 'style100']:
        from data_loaders.humanml.data.dataset import HumanML3D_Style
        return HumanML3D_Style
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pairs=False):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    if name in ['style100', 'bandai-1', 'bandai-2']:
        if not pairs:
            return t2m_style_collate
        else:
            return style_pairs_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', pairs=False):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "style100"]:
        # The num_frames will be ignored when used humanML and Kit dataset
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name in ["bandai-1", "bandai-2"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, dataset_name=name, pairs=pairs)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', pairs=False):
    dataset = get_dataset(name, num_frames, split, hml_mode, pairs)
    collate = get_collate_fn(name, hml_mode, pairs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

if __name__ == "__main__":
    data_loder = get_dataset_loader('bandai-2', 32, num_frames=20) 
    print(data_loder.dataset.__len__()) #30245
    for i, (motion, cond) in enumerate(data_loder):
        print(motion.shape)
        cond["sty_y"] = cond.pop("y")
        cond["sty_x"] = motion
        # print(cond.keys()) 
        # print(cond)

        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        from data_loaders.humanml.utils.plot_script import plot_3d_array
        import data_loaders.humanml.utils.paramUtil as paramUtil
        import imageio
        import os
        import numpy as np

        motion = data_loder.dataset.style_dataset.inv_transform(motion.cpu().permute(0, 2, 3, 1)).float()
        motion = recover_from_ric(motion, 22) # B 1 T J 3 
        motion = motion.view(-1, *motion.shape[2:]).permute(0, 2, 3, 1)  # B J 3 T 
        motion = motion.cpu().numpy().transpose(0, 3, 1, 2)
        motion_array = plot_3d_array([motion[0][:cond["sty_y"]["lengths"][0]], None, paramUtil.t2m_kinematic_chain, "bandai-dataset-loader-test"])
        imageio.mimsave(os.path.join('./', 'bandai-loader-example.gif'), np.array(motion_array), duration=cond["sty_y"]["lengths"][0]/20)

        exit()
