
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
    CLAP: 0.267

    motion 5s: e18 and e21 looks the best, but might be a bit overfit
    e12, gs3: 'fid_k': 12.921445971613068, 'fid_m': 24.939611892884685, 'div_k': 8.544877247367815, 'div_m': 7.583071551841238
    e15, gs3: 'fid_k': 10.793337536523282, 'fid_m': 15.477890313025043, 'div_k': 8.866393450522118, 'div_m': 7.217119566350136
    e18, gs3: 'fid_k': (12.150440499805114-1.0108535545147974e-06j), 'fid_m': (22.708859243262836-2.5935756483221857e-08j), 'div_k': 8.263081838603073, 'div_m': 8.446190809732103
    e21, gs3: 'fid_k': 18.22769548681839, 'fid_m': 16.428731811593394, 'div_k': 10.538286240042385, 'div_m': 8.152619813429473

    motion2music:
    seed71: 0.9302325581395349 0.8837209302325582

    text2motion
    default: 'fid_k': 9.399568536440668, 'fid_m': 14.34595311292712, 'div_k': 8.721145626354044, 'div_m': 8.473095050731231  Beat align: 0.2526632596256102
    seed11: 'fid_k': 8.005759961413077, 'fid_m': (22.063916396531354-1.3458773653855371e-08j), 'div_k': 9.502469831763486, 'div_m': 9.79012076291153  Beat align: 0.22928657313550185
    seed21: 'fid_k': 13.338196937976363, 'fid_m': (26.870628049694403-2.3999753771876277e-08j), 'div_k': 10.024508288615822, 'div_m': 10.211402362853521 Beat align: 0.22928657313550185
    seed31: 'fid_k': 6.579166959447548, 'fid_m': 27.441034028171543, 'div_k': 9.135667565247685, 'div_m': 9.37393172642592 Beat align: 0.2521137245028993
    seed41: 'fid_k': 9.962895629602869, 'fid_m': (11.922686256653876-8.566203601383508e-09j), 'div_k': 7.612997132020285, 'div_m': 8.448205224113453 Beat align: 0.23968354153145663

    music2motion aist new
    gs4
        default: 'fid_k': 24.597259471308718, 'fid_m': (65.36125819984692-1.3896335701109031e-08j), 'div_k': 11.596673399155101, 'div_m': 10.898799443186872  0.22692302316832413
        random: 'fid_k': 24.597259471308718, 'fid_m': (65.36125819984692-1.3896335701109031e-08j), 'div_k': 11.596673399155101, 'div_m': 10.898799443186872  
        seed11: 'fid_k': 32.421401802145, 'fid_m': (74.79974465507372-1.9278348186426132e-08j), 'div_k': 12.105120992530948, 'div_m': 11.751218937314386  0.23000505921667652
        seed21: 'fid_k': 19.98485146827082, 'fid_m': (74.10202790929132-6.575057411186891e-09j), 'div_k': 10.483741577328788, 'div_m': 11.57724740150692  0.23164901789199654
        seed31: 'fid_k': 19.60760174370637, 'fid_m': (66.64591641235673-2.032788965685277e-08j), 'div_k': 11.446574749158644, 'div_m': 11.102661902066803 0.22474862865726306
        seed41: 'fid_k': 28.024590143366538, 'fid_m': (61.67015850841847-1.7054909344871837e-08j), 'div_k': 11.751614712810653, 'div_m': 11.05107785895357 0.22474862865726306
    gs3
        default: 'fid_k': 15.812657191456978, 'fid_m': (54.06928970806986-2.243918211879271e-08j), 'div_k': 10.090091834232002, 'div_m': 10.508459259967221  0.2148267785871961
        seed12: 'fid_k': 16.053487180150853, 'fid_m': 51.213404778308615, 'div_k': 10.684426744097442, 'div_m': 10.35185954607023  0.2399649409689576
        seed22: 'fid_k': 16.911316662090798, 'fid_m': (59.801347291391494-6.724180564936086e-09j), 'div_k': 10.932261464189331, 'div_m': 10.783203808010601  0.22623119388546956
        seed32: 'fid_k': 15.841353824920603, 'fid_m': 37.47853661579198, 'div_k': 10.100916786598612, 'div_m': 9.344810236416656  0.2409615748196112

    gs2
        default: 'fid_k': 15.624760536143015, 'fid_m': 43.67848131737149, 'div_k': 9.45608334752217, 'div_m': 9.592410583327858  0.2160752163061783 

    gs1
        default: 'fid_k': 23.355578635735412, 'fid_m': 34.935461140950885, 'div_k': 11.057021694073317, 'div_m': 8.76186337314385  0.20909610979297197


## Attempt 13.3: Try with new music4all_ignore

    musiccap: sounds not good for all...
    e12, gs4: 'frechet_distance': 43.5271, 'frechet_audio_distance': 5.3584, 'kl_sigmoid': 6.6753, 'kl_softmax': 2.0042
    e15, gs4: 'frechet_distance': 44.5799, 'frechet_audio_distance': 7.4903, 'kl_sigmoid': 6.5004, 'kl_softmax': 1.9895
    e18, gs4: 'frechet_distance': 45.0807, 'frechet_audio_distance': 6.4447, 'kl_sigmoid': 6.7654, 'kl_softmax': 2.0635
    e21, gs4: 'frechet_distance': 43.6362, 'frechet_audio_distance': 7.0687, 'kl_sigmoid': 7.0275, 'kl_softmax': 2.0871
    
