
# Attempt 8: Mixture of Expert only
Branch from attempt3, and remove all the pretraining and separate codebook. Just adding MoE, and
a slightly different text conditioning (instead of using None as unconditional) in comparison with 
main branch.

    musiccap
    e18, gs4: 'frechet_distance': 45.8688, 'frechet_audio_distance': 6.8478, 'kl_sigmoid': 6.9925, 'kl_softmax': 2.1085
    e24, gs4: 'frechet_distance': 46.2312, 'frechet_audio_distance': 8.7354, 'kl_sigmoid': 7.1881, 'kl_softmax': 2.1206
    e30, gs4: 'frechet_distance': 45.7157, 'frechet_audio_distance': 7.3616, 'kl_sigmoid': 7.1034, 'kl_softmax': 2.1314

    motion 5s:
    e18, gs2: 'fid_k': 10.928846652395606, 'fid_m': 13.483807775808259, 'div_k': 7.4025383572723955, 'div_m': 7.1287259314796465
         gs3: 'fid_k': 10.723358063397384, 'fid_m': (11.641404402818353-2.666403214230605e-08j), 'div_k': 7.924166038176554, 'div_m': 6.73196192185084,
         gs4: 'fid_k': 10.031844493577552, 'fid_m': 13.778549700123854, 'div_k': 7.332399139818273, 'div_m': 6.2658486151429456