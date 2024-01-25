import random
import argparse
import os
from os.path import join as pjoin
import soundfile as sf
import librosa
import subprocess
from pytorch_lightning import seed_everything
import numpy as np

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


# Test music-to-motion on AIST++ dataset.
# The data and split are downloaded from https://github.com/L-YeZhu/D2M-GAN (since I cannot find elsewhere
# the aligned music and motion of AIST++)


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
        "--aist_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/aist_plusplus_final"
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
        "-g",
        "--use_genre",
        type=bool,
        required=False,
        default=False,
        help="Whether to add genre condition on motion according to the music filename",
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
    aist_dir = args.aist_dir
    motion_dir = args.motion_dir

    music_id_list = []
    with open(pjoin(aist_dir, 'aist_audio_test_segment.txt'), 'r') as f:
        for line in f.readlines():
            music_id_list.append(line.strip())
    print('number of testing data:', len(music_id_list))

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']
    aist_genre_map = {
        'gBR': 'break',
        'gPO': 'pop',
        'gLO': 'lock',
        'gMH': 'middle hip-hop',
        'gLH': 'LA style hip-hop',
        'gHO': 'house',
        'gWA': 'waack',
        'gKR': 'krump',
        'gJS': 'street jazz',
        'gJB': 'ballet jazz'
    }

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    total_num = len(music_id_list)
    count = 0
    while count < total_num:
        music_id = music_id_list[count].split('/')[-1].split('.')[0]
        print(music_id)

        music_path = pjoin(aist_dir, music_id_list[count][1:])
        waveform, _ = librosa.load(music_path, sr=32000)
        waveform = waveform[None, None, ...]

        music_description = 'This is a pop dance music, with fast tempo and strong intensity.'
        # generate some random motion captions
        # use aist style prompts
        if args.use_genre:
            genre_id = music_id.split('_')[0]
            genre = aist_genre_map[genre_id]
        else:
            genre = random.choice(aist_genres)
        motion_description = f'The style of the dance is {genre}.'

        text_description = '<music_prompt_start> ' + music_description.capitalize() + \
                           ' <music_prompt_end> <motion_prompt_start> ' + \
                           motion_description.capitalize() + ' <motion_prompt_end>'

        print(f'waveform: {waveform.shape}')
        motion_gen = model.generate_motion_from_music(
            waveform=waveform,
            text_description=[text_description],
            conditional_guidance_scale=guidance_scale
        )
        waveform_gen = waveform.squeeze()
        joint_gen = motion_gen['joint'][0]
        print(f'waveform gen: {waveform_gen.shape}, joint_gen: {joint_gen.shape}')

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
                fps=model.motion_vqvae.motion_encoder.input_fps, radius=4
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
