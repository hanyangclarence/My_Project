
# Attempt 11: Finetune musicgen
This branch removes motion part, and only generate music

    e6, gs4: 'frechet_distance': 44.1358, 'frechet_audio_distance': 6.1041, 'kl_sigmoid': 6.7596, 'kl_softmax': 1.9815
    e12, gs4: 'frechet_distance': 44.4309, 'frechet_audio_distance': 7.6231, 'kl_sigmoid': 6.807, 'kl_softmax': 2.002
    e18, gs4: 'frechet_distance': 45.3321, 'frechet_audio_distance': 6.4107, 'kl_sigmoid': 6.8005, 'kl_softmax': 2.0655
    e24, gs4: 'frechet_distance': 46.8883, 'frechet_audio_distance': 7.3609, 'kl_sigmoid': 7.0472, 'kl_softmax': 2.1119

# Attempt 11.2: Finetune with new music4all_ignore
