from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_style_collate

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

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    if name in ['style100', 'bandai-1', 'bandai-2']:
        return t2m_style_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "style100"]:
        # The num_frames will be ignored when used humanML and Kit dataset
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name in ["bandai-1", "bandai-2"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, dataset_name=name)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

if __name__ == "__main__":
    data_loder = get_dataset_loader('style100', 32, num_frames=20) 
    print(data_loder.dataset.__len__()) #30245
    for i, (motion, cond) in enumerate(data_loder):
        print(motion.shape)
        cond["sty_y"] = cond.pop("y")
        cond["sty_x"] = motion
        print(cond.keys()) 
        print(cond)
        exit()
