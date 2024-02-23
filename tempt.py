from unimumo.alignment import visual_beat, interpolation
from unimumo.motion import motion_process
import numpy as np
import torch
import librosa

def calculate_alignment(waveform, sr, motion, fps):
    assert waveform.shape[-1] / sr == motion.shape[0] / fps, f'{waveform.shape}, {sr}, {motion.shape}, {fps}'

    rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
    skel = rec_ric_data.squeeze().numpy()
    directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
    peakinds, peakvals = visual_beat.get_candid_peaks(vimpact, sampling_rate=fps)
    tempo_bpms, result = visual_beat.getVisualTempogram(vimpact, window_length=4, sampling_rate=fps)
    visual_beats = visual_beat.find_optimal_paths(
        list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=fps
    )
    visual_beats = [beat[0] for beat in visual_beats[0]]
    visual_beats_time = [b / fps for b in visual_beats]

    tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr)
    music_beat_time = librosa.frames_to_time(beats, sr=sr)
    print('here')


motion = np.load('data/motion/test/joint_vecs/gKR_sBM_cAll_d30_mKR5_ch09.npy')
fps = 60
waveform, sr = librosa.load('data/music/audio.mp3', sr=32000)
calculate_alignment(waveform, sr, motion, fps)