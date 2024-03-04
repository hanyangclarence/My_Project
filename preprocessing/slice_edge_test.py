import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        print(f'{out_dir}/{file_name}_slice{idx}.wav, {audio_slice.shape}')
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = np.load(motion_file)
    motion_length = motion.shape[0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= motion_length - window and slice_count < num_slices:
        motion_slice = motion[start_idx : start_idx + window]
        np.save(f"{out_dir}/{file_name}_slice{slice_count}.npy", motion_slice)
        print(f'{out_dir}/{file_name}_slice{slice_count}.npy', motion_slice.shape)
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


if __name__ == '__main__':
    slice_aistpp('data/motion/edge_test/motion', 'data/motion/edge_test/music')
