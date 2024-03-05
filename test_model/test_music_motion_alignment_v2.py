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


def calc_motion_beat(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 22, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    motion_beats = np.asarray(motion_beats[0])
    print(f'keypoint: {keypoints.shape}, motion_beat: {motion_beats}')
    return motion_beats


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


def feature_to_joint(feature_vec):
    feature_vec = normalize_motion(feature_vec)
    feature_vec = torch.Tensor(feature_vec)
    return motion_vec_to_joint(feature_vec, motion_mean=motion_mean, motion_std=motion_std)


def normalize_motion(vec):
    return (vec - motion_mean) / motion_std


if __name__ == '__main__':
    seed_everything(2023)

    # music_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/aist_full/aist_plusplus_final/wav'
    music_dir = 'data/music/audios'
    motion_meta_dir = 'data/motion'


    test_time = 300
    duration = 10
    sr = 32000




    prev_score = []
    after_score = []

    music_id_list = os.listdir(music_dir)
    motion_id_list = os.listdir(pjoin(motion_meta_dir, 'train', 'joint_vecs')) + os.listdir(pjoin(motion_meta_dir, 'test', 'joint_vecs')) + os.listdir(pjoin(motion_meta_dir, 'val', 'joint_vecs'))
    motion_id_list = [s for s in motion_id_list if s.startswith('g')]

    print(f'Total music: {len(music_id_list)}, total motion: {len(motion_id_list)}')

    motion_mean = np.load(pjoin(motion_meta_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_meta_dir, 'Std.npy'))

    count = 0
    while count < test_time:
        music_id = random.choice(music_id_list)
        motion_id = random.choice(motion_id_list)

        waveform, _ = librosa.load(pjoin(music_dir, music_id), sr=sr)

        music_length = waveform.shape[0]
        music_target_length = duration * sr
        if music_length > music_target_length:
            start_idx = random.randint(0, music_length - music_target_length)
            waveform = waveform[start_idx:start_idx + music_target_length]
        else:
            print(f'!!!, {music_length}, {music_id}')

        music_beat = detect_music_beat(waveform, motion_fps=60)

        motion_path = pjoin(motion_meta_dir, 'train', 'joint_vecs', motion_id)
        if not os.path.exists(motion_path):
            motion_path = pjoin(motion_meta_dir, 'test', 'joint_vecs', motion_id)
        if not os.path.exists(motion_path):
            motion_path = pjoin(motion_meta_dir, 'val', 'joint_vecs', motion_id)
        assert os.path.exists(motion_path), motion_path

        motion = np.load(motion_path)

        # pad to similar length
        motion_length = motion.shape[0]
        motion_target_length = duration * 60
        if motion_target_length // motion_length < 1:
            start_idx = random.randint(0, motion_length - motion_target_length)
            motion = motion[start_idx:start_idx + motion_target_length]
        elif motion_target_length // motion_length == 1:
            pass
        else:
            max_repeat = motion_target_length // motion_length + 1
            motion = np.tile(motion, (max_repeat, 1))

        # Calculate alignment score before alignment
        joint = feature_to_joint(motion)
        motion_beat_prev = calc_motion_beat(joint)

        prev_alignment_score = beat_alignment(music_beat, motion_beat_prev)
        prev_score.append(prev_alignment_score)

        # Do alignment
        mbeat = (np.rint(music_beat)).astype(int)

        try:
            # get motion visual beats
            rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
            skel = rec_ric_data.squeeze().numpy()
            directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
            peakinds, peakvals = visual_beat.get_candid_peaks(vimpact, sampling_rate=60)
            tempo_bpms, result = visual_beat.getVisualTempogram(vimpact, window_length=4, sampling_rate=60)
            visual_beats = visual_beat.find_optimal_paths(
                list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=60
            )
            # turn visual beats into binary
            vbeats = np.zeros((skel.shape[0]))
            if len(visual_beats) != 0:
                for beat in visual_beats[0]:
                    idx = beat[0]
                    vbeats[idx] = 1
        except Exception as e:
            print(e)
            continue

        # turn music beats also into binary
        mbeats = np.zeros(motion_target_length)
        for beat in mbeat:
            if beat < len(mbeats):
                mbeats[beat] = 1

        try:
            alignment = dtw(vbeats, mbeats, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "d"))
            wq = warp(alignment, index_reference=False)
            final_motion = interpolation.interp(motion, wq)
        except Exception as e:  # if alignment fails, try a new one
            print(e)
            continue

        joint_after = feature_to_joint(final_motion)
        motion_beat_after = calc_motion_beat(joint_after)

        after_alignment_score = beat_alignment(music_beat, motion_beat_after)
        after_score.append(after_alignment_score)

        print(f'{count}, {prev_alignment_score}, {after_alignment_score}')

        count += 1

    print(sum(prev_score) / len(prev_score))
    print(sum(after_score) / len(after_score))





















