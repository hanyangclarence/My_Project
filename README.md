
# Attempt 13: Rerun attempt 1 with separate conditioning
Both MoE and separate codebook are applied. Train directly from scratch.
In attempt1, there are still some bugs in conditioning, but still gets
a good results. Now I run again with a fixed conditioning.

Also, I improved the conditioning part. In training, I removed the conditioning
choices of music2motion and motion2music, and use the built-in cfg dropout to 
achieve this.

    musiccap
    e12, gs4: 'frechet_distance': 43.4326, 'frechet_audio_distance': 6.635, 'kl_sigmoid': 7.0594, 'kl_softmax': 2.0629 (but actually this sounds not good...)
    e15, gs4: 'frechet_distance': 48.4096, 'frechet_audio_distance': 7.5585, 'kl_sigmoid': 6.9251, 'kl_softmax': 2.0989
    e18, gs4: 'frechet_distance': 47.2559, 'frechet_audio_distance': 7.3662, 'kl_sigmoid': 7.0035, 'kl_softmax': 2.0931
    e21, gs4: 'frechet_distance': 45.2028, 'frechet_audio_distance': 8.2851, 'kl_sigmoid': 6.9348, 'kl_softmax': 2.0328

    motion 5s (these all looks not bad, even for e12)
    e12, gs3: 'fid_k': 36.326847193288415, 'fid_m': (35.43533170068264-1.711720717605751e-08j), 'div_k': 11.137739611485332, 'div_m': 8.760789694770125
    e15, gs3: 'fid_k': 22.002664721670158, 'fid_m': 11.238396723825929, 'div_k': 5.916520277704948, 'div_m': 6.364599623358502
    e18, gs3: 'fid_k': 7.426635893096616, 'fid_m': (20.346065855620857-2.9180680199539163e-08j), 'div_k': 8.330986392854182, 'div_m': 8.424083621071334
    e21, gs3: 'fid_k': 10.310515261638443, 'fid_m': (24.540422924567324-1.8279451559923813e-08j), 'div_k': 8.838213407794635, 'div_m': 8.300819460123014


## Attempt 13.2: Try more motion weight
It seems that music converges faster than motion.
Now I change motion weight to 0.15


## Attempt 13.3: Try with new music4all_ignore
