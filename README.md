
# Attempt 8: Mixture of Expert only
Branch from attempt3, and remove all the pretraining and separate codebook. Just adding MoE, and
a slightly different text conditioning (instead of using None as unconditional) in comparison with 
main branch.

    e18, gs4: 'frechet_distance': 45.8688, 'frechet_audio_distance': 6.8478, 'kl_sigmoid': 6.9925, 'kl_softmax': 2.1085
    e24, gs4: 'frechet_distance': 46.2312, 'frechet_audio_distance': 8.7354, 'kl_sigmoid': 7.1881, 'kl_softmax': 2.1206
    e30, gs4: 'frechet_distance': 45.7157, 'frechet_audio_distance': 7.3616, 'kl_sigmoid': 7.1034, 'kl_softmax': 2.1314