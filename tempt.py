import torch

path_og = '../My_Tempt_Repo/pretrained/musicgen_small.bin'
path_trained = 'training_logs/2024-02-02T09-28-27_exp_3_2_stage1/checkpoints/e_18.ckpt'

og_ckpt = torch.load(path_og, map_location='cpu')
new_ckpt = torch.load(path_trained, map_location='cpu')

og_ckpt = og_ckpt['best_state']
new_ckpt = new_ckpt['state_dict']

for name, param in og_ckpt.items():
    new_name = 'model.' + name
    if new_name in new_ckpt.keys():
        print(f"{new_name}, {torch.sum(param - new_ckpt[new_name])}")
    else:
        print(f'{name}, {new_name}')

