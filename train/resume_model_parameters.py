import torch 
from utils.parser_util import style_trans_evaluation_args, eval_style_diffusion_module_args, eval_inpainting_module_args, eval_inpainting_style_args
from utils.model_util import creat_style_trans_module, creat_stylediffuse_and_diffusion
from model.mdm import ControlMDM, StyleDiffusion
from model.mdm_forstyledataset import StyleDiffusion as StyleDiffusion1
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion

def reuse_controlnet():
    pretrained_model = './save/my_motion_enc_512_finetune_encmdm_bandai-2_addtextloss/model000680161.pt'

    output_path = './save/my_style_transfer_module_zero/model_pretrained.pt'

    args = style_trans_evaluation_args()

    transfer_model = creat_style_trans_module(args, None)

    def get_node_name(name):
        control = False
        if 'control_' in name:
            name = 'mdm_model.' + name[len('control_'):]
            control = True
            return control, name
        elif 'seqTransEncoder' in name:
            name = 'mdm_model.' + name
            return control, name
        else:
            return control, name

    pretrained_parameters = torch.load(pretrained_model)

    scratch_dict = transfer_model.state_dict()

    target_dict = {}

    for k in scratch_dict.keys():
        is_control, name = get_node_name(k)
        copy_k = name
        if copy_k in pretrained_parameters:
            target_dict[k] = pretrained_parameters[copy_k].clone()
            print(f'These weights are reused: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()
            
    transfer_model.load_state_dict(target_dict, strict=True)
    torch.save(transfer_model.state_dict(), output_path)
    print('Done.')


def stylediffusion():
    pretrained_model = './save/my_motion_enc_512_finetune_encmdm_bandai-2_addtextloss/model000680161.pt'
    
    output_path = './save/my_style_diffusion/model_pretrained.pt'

    args = eval_style_diffusion_module_args()

    diffusion_model, _ = creat_stylediffuse_and_diffusion(args)
    
    def get_node_name(name):
        control = False
        if 'seqTransEncoder' in name:
            name = 'mdm_model.' + name
            return control, name
        else:
            return control, name

    pretrained_parameters = torch.load(pretrained_model)

    scratch_dict = diffusion_model.state_dict()

    target_dict = {}

    for k in scratch_dict.keys():
        is_control, name = get_node_name(k)
        copy_k = name
        if copy_k in pretrained_parameters:
            target_dict[k] = pretrained_parameters[copy_k].clone()
            print(f'These weights are reused: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'not reused weights: {k}')
            
    diffusion_model.load_state_dict(target_dict, strict=True)
    torch.save(diffusion_model.state_dict(), output_path)
    print('Done.')


def inpainting_motion():
    pretrained_model = './save/my_humanml_trans_enc_512/model000600161.pt'
    
    output_path = './save/my_inpainting_model/model_pretrained.pt'

    args = eval_inpainting_module_args()
    

    diffusion_model, _ = creat_stylediffuse_and_diffusion(args, ModelClass=ControlMDM, DiffusionClass=InpaintingGaussianDiffusion)
    
    def get_node_name(name):
        if 'seqTransEncoder' in name:
            return name

    pretrained_parameters = torch.load(pretrained_model)

    scratch_dict = diffusion_model.state_dict()

    target_dict = {}

    for k in scratch_dict.keys():
        name = get_node_name(k)
        copy_k = name
        if copy_k in pretrained_parameters:
            target_dict[k] = pretrained_parameters[copy_k].clone()
            print(f'These weights are reused: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'not reused weights: {k}')
            
    diffusion_model.load_state_dict(target_dict, strict=True)
    torch.save(diffusion_model.state_dict(), output_path)
    print('Done.')

def inpainting_style():
    pretrained_model = './save/my_inpainting_model/model000080047.pt'
    output_path = './save/my_inpainting_style_model/model_pretrained.pt'
    
    args = eval_inpainting_style_args()

    diffusion_model, _ = creat_stylediffuse_and_diffusion(args, ModelClass=StyleDiffusion, DiffusionClass=InpaintingGaussianDiffusion)
    
    def get_node_name(name):
        if 'seqTransEncoder' in name:
            return name

    pretrained_parameters = torch.load(pretrained_model)

    scratch_dict = diffusion_model.state_dict()

    target_dict = {}

    for k in scratch_dict.keys():
        name = get_node_name(k)
        copy_k = name
        if copy_k in pretrained_parameters:
            target_dict[k] = pretrained_parameters[copy_k].clone()
            print(f'These weights are reused: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'not reused weights: {k}')
            
    diffusion_model.load_state_dict(target_dict, strict=True)
    torch.save(diffusion_model.state_dict(), output_path)
    print('Done.')


def inpainting_style_onstyledataset():
    
    # pretrained_model = './save_bandai/my_inpainting_model_new/model000100000.pt'
    # output_path = './save_bandai/my_inpainting_style_model_regularization_ddim20/model_pretrained.pt'
    # pretrained_model = './save_stylexia/my_inpainting_model/model000050000.pt'
    # output_path = './save_stylexia/my_inpainting_style_model/model_pretrained.pt'
    pretrained_model = './save_bandai_train/my_inpainting_model_new/model000100000.pt'
    output_path = './save_bandai_train/my_inpainting_style_model/model_pretrained.pt'
    

    args = eval_inpainting_style_args()

    diffusion_model, _ = creat_stylediffuse_and_diffusion(args, ModelClass=StyleDiffusion1, DiffusionClass=InpaintingGaussianDiffusion)
    
    def get_node_name(name):
        if 'seqTransEncoder' in name:
            return name

    pretrained_parameters = torch.load(pretrained_model)

    scratch_dict = diffusion_model.state_dict()

    target_dict = {}

    for k in scratch_dict.keys():
        if k.startswith('motion_enc.'):
            continue
        name = get_node_name(k)
        copy_k = name
        if copy_k in pretrained_parameters:
            target_dict[k] = pretrained_parameters[copy_k].clone()
            print(f'These weights are reused: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'not reused weights: {k}')
            
    # diffusion_model.load_state_dict(target_dict, strict=True)
    torch.save(target_dict, output_path)
    print('Done.')


if __name__ == "__main__":
    # stylediffusion()
    # inpainting_motion()
    # inpainting_style()
    inpainting_style_onstyledataset()