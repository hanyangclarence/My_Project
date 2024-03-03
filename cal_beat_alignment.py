import numpy as np
import torch
import librosa
import os
from os.path import join as pjoin

from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema


def calc_motion_beat(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 22, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    motion_beats = np.asarray(motion_beats[0])
    print(f'keypoint: {keypoints.shape}, motion_beat: {motion_beats.shape}', end=', ')
    return motion_beats


def get_extracted_music_beat(music_id, motion_fps):
    feature_path = pjoin(music_beat_dir, f'{music_id}.pth')
    music_beat_all = torch.load(feature_path)['beat']
    music_beat_all = music_beat_all / 32000 * motion_fps  # change to the time scale of motion

    music_beat = []
    for b in music_beat_all:
        if b < motion_fps * duration:
            music_beat.append(b)

    music_beat = np.asarray(music_beat)
    print(f'music_beat: {music_beat.shape}')
    return music_beat

def detect_music_beat(music_id, motion_fps):
    audio_path = pjoin(music_dir, music_id + '.mp3')
    waveform, sr = librosa.load(audio_path, sr=32000)
    tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times *= motion_fps
    print(f'music_beat: {beat_times.shape}')
    return beat_times


def beat_alignment(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)


if __name__ == '__main__':
    music_beat_dir = 'data/music/music4all_beat'
    duration = 5

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--joint_dir",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "-m",
        "--music_dir",
        type=str,
        default=None,
        help="",
    )
    args = parser.parse_args()

    joint_dir = args.joint_dir
    music_dir = args.music_dir
    music_id_list = os.listdir(joint_dir)
    music_id_list = [s.split('.')[0] for s in music_id_list if s.endswith('.npy')]
    print(f'Total number of data to test: {len(music_id_list)}')
    total_num = len(music_id_list)

    beat_align_scores = []
    for music_id in music_id_list:
        joint_path = pjoin(joint_dir, f'{music_id}.npy')
        try:
            joint = np.load(joint_path)
            motion_beat = calc_motion_beat(joint)
            if music_dir is None:
                music_beat = get_extracted_music_beat(music_id, 60)
            else:
                music_beat = detect_music_beat(music_id, 60)
            if len(music_beat) == 0:
                print(f'Error! music beat not detected')
                continue
        except Exception as e:
            print(e)
            continue
        print(f'music_beat: ', music_beat)
        print(f'motion_beat: ', motion_beat)
        score = beat_alignment(music_beat, motion_beat)
        print(f'score: {score}')
        beat_align_scores.append(score)

    final_score = sum(beat_align_scores) / len(beat_align_scores)
    print(f'Final: {final_score}')
