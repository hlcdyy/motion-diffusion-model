import torch 
from utils.parser_util import style_trans_evaluation_args 
from utils.model_util import creat_style_trans_module

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
