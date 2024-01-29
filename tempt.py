import os
import librosa
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
from unimumo.audio.audiocraft_.models.builders import get_compression_model


file_list = os.listdir('../Data_analysis/audios')[:20]
path = '../My_Tempt_Repo/pretrained/music_vqvae.bin'
pkg = torch.load(path, map_location='cpu')
cfg = OmegaConf.create(pkg['xp.cfg'])
model = get_compression_model(cfg)
model.load_state_dict(pkg['best_state'])
model = model.cuda()

with open('rank_recon_loss.txt', 'w') as f:
    results = {}
    for i, file in enumerate(file_list):
        waveform, sr = librosa.load(os.path.join('../Data_analysis/audios', file), sr=32000)

        waveform = torch.tensor(waveform)[None, None, ...].cuda()
        waveform = waveform[..., :32000 * 6]
        recon = model.forward(waveform).x

        recon_loss = torch.nn.functional.mse_loss(waveform, recon)

        results[file] = recon_loss
        print(f'{i+1} {file}: loss: {recon_loss}')

    sorted_result = dict(sorted(results.items(), key=lambda x: x[1]))
    for k, v in sorted_result.items():
        f.write(f'{k}\t{v}\n')

    values = sorted_result.values()
    plt.hist(values, bins=80)
    plt.savefig('recon_loss_histogram.png')

# -25 < loudness < -7
