import random
import argparse
import os
from os.path import join as pjoin
import soundfile as sf
import librosa
import subprocess
from pytorch_lightning import seed_everything
import numpy as np
import json

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_music2motion_aist",
    )

    parser.add_argument(
        "--music_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="data/music/audios"
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion',
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=3.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
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
        "--start",
        type=float,
        required=False,
        default=0.,
        help="start ratio",
    )

    parser.add_argument(
        "--end",
        type=float,
        required=False,
        default=1.,
        help="end ratio",
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    joint_save_path = pjoin(save_path, 'joint')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(joint_save_path, exist_ok=True)
    guidance_scale = args.guidance_scale
    music_dir = args.music_dir
    motion_dir = args.motion_dir
    duration = 5
    num_segment = 1

    # load random motion descriptions
    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']
    music_captions = json.load(open('data/music/music4all_captions_mullama.json', 'r'))

    music_id_list = os.listdir(music_dir)
    music_id_list = [s.split('.')[0] for s in music_id_list]
    music_id_list = [s for s in music_id_list if s in music_captions.keys()]
    music_id_list = music_id_list[:300]
    print('number of testing data:', len(music_id_list))

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    total_num = len(music_id_list)
    start_idx = int(args.start * total_num)
    end_idx = int(args.end * total_num)
    count = start_idx
    print(f'start: {count}, end: {end_idx}')
    while count < end_idx:
        music_path = pjoin(music_dir, music_id_list[count] + '.wav')
        if not os.path.exists(music_path):
            music_path = pjoin(music_dir, music_id_list[count] + '.mp3')
        if not os.path.exists(music_path):
            print(f'{music_path} does not exist!')
            count += 1
            continue
        waveform, _ = librosa.load(music_path, sr=32000)
        target_length = duration * 32000 * num_segment
        start_idx = random.randint(0, waveform.shape[-1] - target_length - 1)
        waveform = waveform[start_idx:start_idx + target_length]
        waveform = waveform.reshape((num_segment, 1, -1))

        music_description = music_captions[music_id_list[count]]
        # generate some random motion captions
        genre = random.choice(aist_genres)
        motion_description = f'The style of the dance is {genre}.'

        text_description = '<music_prompt_start> ' + music_description.capitalize() + \
                           ' <music_prompt_end> <motion_prompt_start> ' + \
                           motion_description.capitalize() + ' <motion_prompt_end>'

        print(f'waveform: {waveform.shape}, caption: {text_description}')
        motion_gen = model.generate_motion_from_music(
            waveform=waveform,
            text_description=[text_description] * num_segment,
            conditional_guidance_scale=guidance_scale
        )
        waveform_gen = waveform.reshape(-1)
        joint_gen = motion_gen['joint']
        joint_gen = joint_gen.reshape((-1, joint_gen.shape[-2], joint_gen.shape[-1]))
        print(f'waveform gen: {waveform_gen.shape}, joint_gen: {joint_gen.shape}')

        music_id = music_id_list[count].split('/')[-1].split('.')[0]
        print(music_id)

        music_filename = "%s.mp3" % music_id
        music_path = os.path.join(music_save_path, music_filename)
        try:
            sf.write(music_path, waveform_gen, 32000)
        except Exception as e:
            print(e)
            count += 1
            continue

        motion_filename = "%s.mp4" % music_id
        motion_path = pjoin(motion_save_path, motion_filename)
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                fps=model.motion_fps, radius=4
            )
        except Exception as e:
            print(e)
            count += 1
            continue

        video_filename = "%s.mp4" % music_id
        video_path = pjoin(video_save_path, video_filename)
        try:
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)
        except Exception as e:
            print(e)
            count += 1
            continue

        joint_filename = "%s.npy" % music_id
        joint_path = pjoin(joint_save_path, joint_filename)
        np.save(joint_path, joint_gen)

        count += 1