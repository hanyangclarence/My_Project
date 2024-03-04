import librosa
import soundfile
import numpy as np
import os
from os.path import join as pjoin

test_data_id_list = [
    'gLH_sBM_cAll_d17_mLH4_ch02',
    'gLH_sBM_cAll_d18_mLH4_ch02',
    'gKR_sBM_cAll_d30_mKR2_ch02',
    'gKR_sBM_cAll_d28_mKR2_ch02',
    'gBR_sBM_cAll_d04_mBR0_ch02',
    'gBR_sBM_cAll_d05_mBR0_ch02',
    'gLO_sBM_cAll_d13_mLO2_ch02',
    'gLO_sBM_cAll_d15_mLO2_ch02',
    'gJB_sBM_cAll_d08_mJB5_ch02',
    'gJB_sBM_cAll_d09_mJB5_ch02',
    'gWA_sBM_cAll_d26_mWA0_ch02',
    'gWA_sBM_cAll_d25_mWA0_ch02',
    'gJS_sBM_cAll_d03_mJS3_ch02',
    'gJS_sBM_cAll_d01_mJS3_ch02',
    'gMH_sBM_cAll_d24_mMH3_ch02',
    'gMH_sBM_cAll_d22_mMH3_ch02',
    'gHO_sBM_cAll_d20_mHO5_ch02',
    'gHO_sBM_cAll_d21_mHO5_ch02',
    'gPO_sBM_cAll_d10_mPO1_ch02',
    'gPO_sBM_cAll_d11_mPO1_ch02'
]

motion_data_meta_dir = 'data/motion'
full_music_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/aist_full/aist_plusplus_final/wav'
save_dir = 'data/motion/edge_test'
motion_save_dir = pjoin(save_dir, 'motion')
music_save_dir = pjoin(save_dir, 'music')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(motion_save_dir, exist_ok=True)
os.makedirs(music_save_dir, exist_ok=True)

for motion_id in test_data_id_list:
    motion_path = pjoin(motion_data_meta_dir, 'train', 'joint_vecs', motion_id + '.npy')
    if not os.path.exists(motion_path):
        motion_path = pjoin(motion_data_meta_dir, 'val', 'joint_vecs', motion_id + '.npy')
    if not os.path.exists(motion_path):
        motion_path = pjoin(motion_data_meta_dir, 'test', 'joint_vecs', motion_id + '.npy')
    assert os.path.exists(motion_path), motion_path

    motion = np.load(motion_path)
    motion_length = motion.shape[0]
    duration = motion_length / 60

    music_id = motion_id.split('_')[4]
    music_path = pjoin(full_music_dir, music_id + '.wav')
    wave, _ = librosa.load(music_path, sr=32000)
    sr = 32000

    wave_cropped = wave[:int(duration * sr)]

    soundfile.write(pjoin(music_save_dir, motion_id + '.wav'), wave_cropped, samplerate=sr)
    np.save(pjoin(motion_save_dir, motion_id + '.npy'), motion)
    print(f"{motion_id}, {motion.shape}, {wave_cropped.shape}, {pjoin(motion_save_dir, motion_id + '.npy')}, {pjoin(music_save_dir, motion_id + '.wav')}")



