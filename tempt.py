from unimumo.alignment import visual_beat, interpolation
from unimumo.motion import motion_process
import numpy as np
import torch
import librosa

motion = np.load('data/motion/test/joint_vecs/gKR_sBM_cAll_d30_mKR5_ch09.npy')
fps = 60

rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
skel = rec_ric_data.squeeze().numpy()
directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
peakinds, peakvals = visual_beat.get_candid_peaks(vimpact, sampling_rate=fps)
tempo_bpms, result = visual_beat.getVisualTempogram(vimpact, window_length=4, sampling_rate=fps)
visual_beats = visual_beat.find_optimal_paths(
    list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=fps
)

visual_beat_idx = [beat[0] for beat in visual_beats[0]]

waveform, sr = librosa.load('data/music/audio.mp3', sr=32000)
tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

print('here')
