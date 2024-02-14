
# Attempt 9: Separate music and motion conditioning

    musiccap
    e12, gs4: 'frechet_distance': 42.8635, 'frechet_audio_distance': 7.3433, 'kl_sigmoid': 7.201, 'kl_softmax': 2.0202
    e18, gs4: 'frechet_distance': 43.6681, 'frechet_audio_distance': 6.9962, 'kl_sigmoid': 6.8329, 'kl_softmax': 2.0487
    e24, gs4: 'frechet_distance': 46.0443, 'frechet_audio_distance': 7.0963, 'kl_sigmoid': 6.9037, 'kl_softmax': 2.0933
    e30, gs4:

    musiccap with music captions only
    e12, gs4: 'frechet_distance': 43.0303, 'frechet_audio_distance': 7.3304, 'kl_sigmoid': 7.1479, 'kl_softmax': 2.0062
    e18, gs4: 'frechet_distance': 43.912, 'frechet_audio_distance': 7.0259, 'kl_sigmoid': 6.861, 'kl_softmax': 2.0599
    e24, gs4: 'frechet_distance': 46.0804, 'frechet_audio_distance': 7.0329, 'kl_sigmoid': 6.9196, 'kl_softmax': 2.0904
    e30, gs4: 'frechet_distance': 44.4852, 'frechet_audio_distance': 7.3429, 'kl_sigmoid': 6.8461, 'kl_softmax': 2.0632

    motion 5s
    e12, gs3: 'fid_k': 19.863799030687815, 'fid_m': 35.730605568571896, 'div_k': 5.544803851719285, 'div_m': 8.594486743678218
    e18, gs3: 'fid_k': 14.283764448565918, 'fid_m': 21.225513503637657, 'div_k': 8.295467562985792, 'div_m': 7.386661992530227
    e24, gs3: 'fid_k': 13.368350321607295, 'fid_m': (13.148379653843826-2.583251388779041e-08j), 'div_k': 7.01786709048344, 'div_m': 7.512311655382647
    e30, gs3: 'fid_k': 25.59991682115694, 'fid_m': 13.827549041884382, 'div_k': 11.612817812032391, 'div_m': 7.864600600212846

I found that previous drop out is duplicated. Now I remove the 
duplicated dropout and run again.

    e12, gs4: 'frechet_distance': 43.4774, 'frechet_audio_distance': 7.3071, 'kl_sigmoid': 7.064, 'kl_softmax': 2.0481
    e15, gs4: 'frechet_distance': 44.4807, 'frechet_audio_distance': 6.2163, 'kl_sigmoid': 6.9078, 'kl_softmax': 2.0458
    e18, gs4: 'frechet_distance': 48.0789, 'frechet_audio_distance': 6.9741, 'kl_sigmoid': 6.9807, 'kl_softmax': 2.087
    e21, gs4: 'frechet_distance': 45.2588, 'frechet_audio_distance': 6.8772, 'kl_sigmoid': 6.8112, 'kl_softmax': 2.0537


## Attempt 9.2: Use new music4all_ignore

