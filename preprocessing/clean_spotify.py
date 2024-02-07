import os
import librosa
import matplotlib.pyplot as plt
import pyloudnorm as pyln

file_list = os.listdir('/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/spotify_music/audios')

with open('rank_loudness.txt', 'w') as f:
    with open('spotify_ignore.txt', 'w') as f_ignore:
        results = {}
        for i, file in enumerate(file_list):
            waveform, sr = librosa.load(os.path.join('/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/spotify_music/audios', file))
            # normalize: !!!!!!!!!!!!!!!!!!!!
            # waveform = waveform / np.max(np.abs(waveform))

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(waveform)
            loudness = max(loudness, -50.)  # remove -inf

            if loudness < -22:
                f_ignore.write(f"{file.split('.')[0]}\n")

            results[file] = loudness
            print(f'{i+1} {file}: loudness: {loudness}')

        sorted_result = dict(sorted(results.items(), key=lambda x: x[1]))
        for k, v in sorted_result.items():
            f.write(f'{k}\t{v}\n')

        values = sorted_result.values()
        plt.hist(values, bins=80)
        plt.savefig('loudness_histogram.png')

# -25 < loudness < -7

# spotify: -22 or -23
