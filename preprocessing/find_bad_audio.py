import os
import librosa
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import pyloudnorm as pyln
import json
from tqdm import tqdm

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.audio.audiocraft_.models.builders import get_compression_model


if __name__ == '__main__':
    audio_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/data/music/audios'
    music_vqvae_path = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/pretrained/music_vqvae.bin'
    save_dir = 'stat'

    os.makedirs(save_dir, exist_ok=True)

    pkg = torch.load(music_vqvae_path, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    model = get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model = model.cuda()

    ignore_list = []
    file_list = os.listdir(audio_dir)[:1000]
    results_rms = {}
    results_loudness = {}
    results_recon = {}
    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            file = file_list[i]
            waveform, sr = librosa.load(os.path.join(audio_dir, file))

            rms = librosa.feature.rms(y=waveform).mean()
            results_rms[file] = rms

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(waveform)
            loudness = max(loudness, -50.)  # remove -inf
            results_loudness[file] = loudness

            waveform, sr = librosa.load(os.path.join(audio_dir, file), sr=32000)

            waveform = torch.tensor(waveform)[None, None, ...].cuda()
            recon = model.forward(waveform).x
            recon_loss = torch.nn.functional.mse_loss(waveform, recon).item()
            results_recon[file] = recon_loss

            print(f'{i+1} {file}: rms: {rms}, loudness: {loudness}, recon loss: {recon_loss}, ratio: {len(ignore_list)/(i+1)}')

            if recon_loss < 0.0025:
                ignore_list.append(file.split('.')[0])
            elif loudness < -25 or loudness > -7:
                ignore_list.append(file.split('.')[0])
            elif rms < 0.06:
                ignore_list.append(file.split('.')[0])

            if i % 1000 == 0:
                # save results at times
                with open(os.path.join(save_dir, 'ignore.txt'), 'w') as f:
                    for name in ignore_list:
                        f.write(f'{name}\n')

                results_recon = dict(sorted(results_recon.items(), key=lambda x: x[1]))
                results_loudness = dict(sorted(results_loudness.items(), key=lambda x: x[1]))
                results_rms = dict(sorted(results_rms.items(), key=lambda x: x[1]))

                with open(os.path.join(save_dir, 'rank_recon_loss.json'), 'w') as f:
                    json.dump(results_recon, f, indent=4)
                with open(os.path.join(save_dir, 'rank_loudness.json'), 'w') as f:
                    json.dump(results_loudness, f, indent=4)
                with open(os.path.join(save_dir, 'rank_rms.json'), 'w') as f:
                    json.dump(results_rms, f, indent=4)

        with open(os.path.join(save_dir, 'ignore.txt'), 'w') as f:
            for name in ignore_list:
                f.write(f'{name}\n')

        results_recon = dict(sorted(results_recon.items(), key=lambda x: x[1]))
        results_loudness = dict(sorted(results_loudness.items(), key=lambda x: x[1]))
        results_rms = dict(sorted(results_rms.items(), key=lambda x: x[1]))

        with open(os.path.join(save_dir, 'rank_recon_loss.json'), 'w') as f:
            json.dump(results_recon, f, indent=4)
        with open(os.path.join(save_dir, 'rank_loudness.json'), 'w') as f:
            json.dump(results_loudness, f, indent=4)
        with open(os.path.join(save_dir, 'rank_rms.json'), 'w') as f:
            json.dump(results_rms, f, indent=4)

        recon_values = results_recon.values()
        plt.hist(recon_values, bins=300)
        plt.savefig(os.path.join(save_dir, 'recon_loss_histogram.png'))

        loudness_value = results_loudness.values()
        plt.hist(loudness_value, bins=300)
        plt.savefig(os.path.join(save_dir, 'loudness_histogram.png'))

        rms_value = results_rms.values()
        plt.hist(rms_value, bins=300)
        plt.savefig(os.path.join(save_dir, 'rms_histogram.png'))

