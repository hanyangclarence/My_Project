import os
import librosa
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
from unimumo.audio.audiocraft_.models.builders import get_compression_model
import random
import soundfile
from pytorch_lightning import seed_everything

seed_everything(42)
file_list = os.listdir('data/music/audios')
file_list = random.sample(file_list, 1000)
path = 'pretrained/music_vqvae.bin'
pkg = torch.load(path, map_location='cpu')
cfg = OmegaConf.create(pkg['xp.cfg'])
model = get_compression_model(cfg)
model.load_state_dict(pkg['best_state'])
model = model.cpu()
os.makedirs('stat', exist_ok=True)
os.makedirs('stat/wav', exist_ok=True)

with open('stat/rank_recon_loss.txt', 'w') as f:
    results = {}
    for i, file in enumerate(file_list):
        waveform, sr = librosa.load(os.path.join('data/music/audios', file), sr=32000)

        waveform = torch.tensor(waveform)[None, None, ...]
        waveform = waveform[..., :32000 * 10]
        recon = model.forward(waveform).x

        recon_loss = torch.nn.functional.mse_loss(waveform, recon)

        results[file] = recon_loss
        print(f'{i+1} {file}: loss: {recon_loss}')
        soundfile.write(os.path.join('stat/wav', file), waveform.squeeze().detach().cpu(), samplerate=32000)

        # waveform = waveform.cpu()
        # del waveform
        # torch.cuda.empty_cache()

    sorted_result = dict(sorted(results.items(), key=lambda x: x[1]))
    for k, v in sorted_result.items():
        f.write(f'{k}\t{v}\n')

    values = sorted_result.values()
    plt.hist(values, bins=80)
    plt.savefig('stat/recon_loss_histogram.png')

