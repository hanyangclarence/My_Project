import random
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
import numpy as np
import torch
import librosa
import os
from os.path import join as pjoin
import torch
from dtw import *
from pytorch_lightning import seed_everything
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.motion.motion_process import motion_vec_to_joint
from unimumo.audio.beat_detection.test_beat_detection import get_music_beat, build_beat_tracker
from unimumo.motion import motion_process
from unimumo.alignment import visual_beat, interpolation


def calc_motion_beat(feature):
    # get motion visual beats
    rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(feature).unsqueeze(0).float(), 22)
    skel = rec_ric_data.squeeze().numpy()
    directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
    peakinds, peakvals = visual_beat.get_candid_peaks(vimpact, sampling_rate=60)
    tempo_bpms, result = visual_beat.getVisualTempogram(vimpact, window_length=4, sampling_rate=60)
    visual_beats = visual_beat.find_optimal_paths(
        list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=60
    )
    motion_beats = []
    # turn visual beats into binary
    if len(visual_beats) != 0:
        for beat in visual_beats[0]:
            idx = beat[0]
            motion_beats.append(idx)

    print(f'keypoint: {feature.shape}, motion_beat: {motion_beats}')
    return np.asarray(motion_beats)


def detect_music_beat(waveform, motion_fps):
    tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times *= motion_fps
    print(f'music_beat: {beat_times}')
    return beat_times


def beat_alignment(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)


def my_beat_alignment(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.min(np.abs(bb - motion_beats))
        # ba += np.exp(-np.min((motion_beats - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)


if __name__ == '__main__':
    seed_everything(2023)

    music_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/data/motion/edge_test/music_sliced/'
    motion_meta_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/data/motion/edge_test/motion_feature_sliced/'

    sr = 32000

    all_scores = []
    linear_scores = []

    music_id_list = os.listdir(music_dir)
    motion_id_list = os.listdir(motion_meta_dir)
    music_id_list = sorted(music_id_list)
    motion_id_list = sorted(motion_id_list)

    print(f'Total music: {len(music_id_list)}, total motion: {len(motion_id_list)}')

    motion_mean = np.load(pjoin('data/motion', 'Mean.npy'))
    motion_std = np.load(pjoin('data/motion', 'Std.npy'))

    count = 0
    while count < len(music_id_list):
        music_id = music_id_list[count]
        motion_id = music_id.split('.')[0] + '.npy'

        waveform, _ = librosa.load(pjoin(music_dir, music_id), sr=sr)

        music_length = waveform.shape[0]

        music_beat = detect_music_beat(waveform, motion_fps=60)
        if len(music_beat) == 0:
            print(f'music beat length = 0! ')
            count += 1
            continue

        motion_path = pjoin(motion_meta_dir, motion_id)
        motion = np.load(motion_path)

        # Calculate alignment score before alignment
        try:
            motion_beat_prev = calc_motion_beat(motion)
        except Exception as e:
            print(e)
            count += 1
            continue

        score = beat_alignment(music_beat, motion_beat_prev)
        all_scores.append(score)
        linear_alignment_score = my_beat_alignment(music_beat, motion_beat_prev)
        linear_scores.append(linear_alignment_score)

        print(f'{count}, {music_id}, {motion_id} {score}, {linear_alignment_score}')

        count += 1

    print(sum(all_scores) / len(all_scores))
    print(sum(linear_scores) / len(linear_scores))





















