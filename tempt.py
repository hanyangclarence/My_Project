import os
import json
import matplotlib.pyplot as plt
import torch

data_dir = 'data/music/music4all_beat'
file_list = os.listdir(data_dir)[:1000]

with open('rank_beat_std.json', 'w') as f:
    results = {}
    for i, file in enumerate(file_list):
        data = torch.load(os.path.join(data_dir, file))
        beat = data['beat']
        interval = beat[1:] - beat[:-1]
        beat_std = interval.float().std()

        results[file] = beat_std
        print(f'{i+1} {file}: std: {beat_std}')

    sorted_result = dict(sorted(results.items(), key=lambda x: x[1]))
    json.dump(sorted_result, f, indent=4)

    values = sorted_result.values()
    plt.hist(values, bins=80)
    plt.savefig('beat_std_histogram.png')

# 0.0025 < loss   (to be further decided)
