
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

    musiccap:
    e12, gs4: 'frechet_distance': 44.5241, 'frechet_audio_distance': 7.2789, 'kl_sigmoid': 6.998, 'kl_softmax': 2.0388
    e15, gs4: (but this sounds pretty good..) 'frechet_distance': 48.1491, 'frechet_audio_distance': 6.8872, 'kl_sigmoid': 6.8832, 'kl_softmax': 2.0731
    e18, gs4: 'frechet_distance': 46.5963, 'frechet_audio_distance': 7.1516, 'kl_sigmoid': 7.0043, 'kl_softmax': 2.0852
    e21, gs4: 'frechet_distance': 45.3224, 'frechet_audio_distance': 5.9036, 'kl_sigmoid': 7.058, 'kl_softmax': 2.0773
    (new metrics) KL_Sigmoid:  6.61594; KL:  1.99115; PSNR: nan SSIM: nan ISc:  1.61455 (0.024696); KID: nan (nan) FD: 38.52773; FAD: 5.93870

    motion 5s: e18 and e21 looks the best, but might be a bit overfit
    e12, gs3: 'fid_k': 12.921445971613068, 'fid_m': 24.939611892884685, 'div_k': 8.544877247367815, 'div_m': 7.583071551841238
    e15, gs3: 'fid_k': 10.793337536523282, 'fid_m': 15.477890313025043, 'div_k': 8.866393450522118, 'div_m': 7.217119566350136
    e18, gs3: 'fid_k': (12.150440499805114-1.0108535545147974e-06j), 'fid_m': (22.708859243262836-2.5935756483221857e-08j), 'div_k': 8.263081838603073, 'div_m': 8.446190809732103
    e21, gs3: 'fid_k': 18.22769548681839, 'fid_m': 16.428731811593394, 'div_k': 10.538286240042385, 'div_m': 8.152619813429473

    motion2music:
    seed71: 0.9302325581395349 0.8837209302325582


## Attempt 13.3: Try with new music4all_ignore

    musiccap: sounds not good for all...
    e12, gs4: 'frechet_distance': 43.5271, 'frechet_audio_distance': 5.3584, 'kl_sigmoid': 6.6753, 'kl_softmax': 2.0042
    e15, gs4: 'frechet_distance': 44.5799, 'frechet_audio_distance': 7.4903, 'kl_sigmoid': 6.5004, 'kl_softmax': 1.9895
    e18, gs4: 'frechet_distance': 45.0807, 'frechet_audio_distance': 6.4447, 'kl_sigmoid': 6.7654, 'kl_softmax': 2.0635
    e21, gs4: 'frechet_distance': 43.6362, 'frechet_audio_distance': 7.0687, 'kl_sigmoid': 7.0275, 'kl_softmax': 2.0871
    
