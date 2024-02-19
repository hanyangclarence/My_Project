import argparse
import json
import os

import numpy as np
import torch
from os.path import join as pjoin
import soundfile as sf
import pandas as pd
import subprocess
import random
from pytorch_lightning import seed_everything

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.motion import skel_animation
from unimumo.motion.utils import kinematic_chain
from unimumo.models import UniMuMo

'''
Load paired music and motion, all made to 10 seconds
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_motion2text_humanml3d",
    )

    parser.add_argument(
        "--music_code_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="data/music/music4all_codes"
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to meta data dir",
        default="data/music",
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion',
    )

    parser.add_argument(
        "--motion_code_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion/aligned_humanml3d_test_motion_code',
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=False,
        default=10,
        help="Generated audio time",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default=None,
        help="load checkpoint",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=10,
        help="end ratio",
    )

    parser.add_argument(
        "--start",
        type=float,
        required=False,
        default=0.0,
        help='the start ratio for this preprocessing'
    )

    parser.add_argument(
        "--end",
        type=float,
        required=False,
        default=1.0,
        help='the end ratio of this preprocessing'
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    feature_263_save_path = pjoin(save_path, 'feature_263')
    feature_22_3_save_path = pjoin(save_path, 'feature_22_3')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(feature_263_save_path, exist_ok=True)
    os.makedirs(feature_22_3_save_path, exist_ok=True)
    batch_size = args.batch_size
    motion_dir = args.motion_dir
    music_code_dir = args.music_code_dir
    motion_code_dir = args.motion_code_dir
    duration = args.duration

    motion_id_list = []
    with open(pjoin(motion_dir, 'humanml3d_test.txt'), 'r') as f:
        for line in f.readlines():
            if os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                motion_id_list.append(line.strip())

    paired_music_motion = os.listdir(motion_code_dir)
    music_data_list = os.listdir(music_code_dir)

    print('number of motion data:', len(motion_id_list), file=sys.stderr)
    print('number of paired motion: ', len(paired_music_motion), file=sys.stderr)

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    result_dict = {}
    count = 0
    with open(pjoin(save_path, f'gen_captions.json'), 'w') as f:
        while count < len(motion_id_list):
            print(f'{count}/{len(motion_id_list)}')
            motion_code_list = []
            music_code_list = []
            motion_id_batch = motion_id_list[count: count + batch_size]

            for motion_id in motion_id_batch:
                # find a paired music code
                selection = [s.split('_!humanml3d_test!_')[0] for s in paired_music_motion if s.split('_!humanml3d_test!_')[1][:-4] == motion_id]
                music_code_id = selection[0]  # just choose the first one
                music_code = torch.load(pjoin(music_code_dir, music_code_id + '.pth'))['codes']  # (1, 4, T)
                motion_code = torch.load(pjoin(motion_code_dir, f'{music_code_id}_!humanml3d_test!_{motion_id}.pth'))  # (4, T)
                motion_code = motion_code[None, ...]  # (1, 4, T)

                # cut first 10 s
                music_code = music_code[:, :, :duration * 50]
                motion_code = motion_code[:, :, :duration * 50]

                music_code_list.append(music_code)
                motion_code_list.append(motion_code)

            music_codes = torch.cat(music_code_list, dim=0).to(device)
            motion_codes = torch.cat(motion_code_list, dim=0).to(device)

            with torch.no_grad():
                print(f'music codes: {music_codes.shape}, motion codes: {motion_codes.shape}')

                batch = {
                    'text': ['<separation>'] * music_codes.shape[0],
                    'music_code': music_codes,
                    'motion_code': motion_codes
                }

                captions = model.music_motion_lm.generate_captions(batch, return_caption_only=True)

                # only log one each time for checking
                waveform_decoded, motion_decoded = model.decode_music_motion(
                    music_codes[0:1], motion_codes[0:1]
                )
                feature_263 = motion_decoded['feature']
                joint = motion_decoded['joint']
                print(f'feature 263: {feature_263.shape}, joint: {joint.shape}')

            os.makedirs(save_path, exist_ok=True)

            music_filename = "%s.mp3" % motion_id_batch[0]
            music_path = os.path.join(music_save_path, music_filename)
            try:
                sf.write(music_path, waveform_decoded.squeeze(), 32000)
            except Exception as e:
                print(e)

            motion_filename = "%s.mp4" % motion_id_batch[0]
            motion_path = pjoin(motion_save_path, motion_filename)
            try:
                skel_animation.plot_3d_motion(
                    motion_path, kinematic_chain, joint, title='None', vbeat=None,
                    fps=model.motion_fps, radius=4
                )
            except Exception as e:
                print(e)

            video_filename = "%s.mp4" % motion_id_batch[0]
            video_path = pjoin(video_save_path, video_filename)
            try:
                subprocess.call(
                    f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                    shell=True)
            except Exception as e:
                print(e)

            feature_263_filename = "%s.npy" % motion_id_batch[0]
            feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
            np.save(feature_263_path, feature_263)

            feature_22_3_filename = "%s.npy" % motion_id_batch[0]
            feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
            np.save(feature_22_3_path, joint)

            # write generated descriptions
            for i in range(len(captions)):
                description = captions[i]

                # split motion description
                description = description.split('<separation>')[-1]
                description = description.replace('The motion is that', '')
                description = description.replace('The dance is that', '')
                description = description.strip().capitalize()

                result_dict[motion_id_batch[i]] = description
                print(f'{motion_id_batch[i]}\t{description}', file=sys.stderr)

            count += batch_size

        json.dump(result_dict, f, indent=4)
