
# Attempt 4: In finetuning, only train part of the parameters

## Attempt 4.1: Train self-attn only

### Attempt 4.1.1: Start with e12

    musiccap:
    e3, gs3 (3571): 'frechet_distance': 48.2074, 'frechet_audio_distance': 7.9447, 'kl_sigmoid': 7.8042, 'kl_softmax': 2.3022
    e6, gs3 (3200): 'frechet_distance': 48.1548, 'frechet_audio_distance': 8.1647, 'kl_sigmoid': 8.0378, 'kl_softmax': 2.3493
    e9, gs3 (5498): 'frechet_distance': 50.2065, 'frechet_audio_distance': 7.9084, 'kl_sigmoid': 7.9244, 'kl_softmax': 2.3403
    ...

### Attempt 4.1.2: Start with e18

## Attempt 4.2: Train self-attn and cross-attn

### Attempt 4.2.1: Start with e12

