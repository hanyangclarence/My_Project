
# Attempt 4: In finetuning, only train part of the parameters

## Attempt 4.1: Train self-attn only

### Attempt 4.1.1: Start with e12

    musiccap:
    e3, gs3:
    e6, gs3:
    e9, gs3:
    ...

### Attempt 4.1.2: Start with e18

## Attempt 4.2: Train self-attn and cross-attn

### Attempt 4.2.1: Start with e12

    musiccap:
    e3, gs3 (4111): 
    e6, gs6 (5489): 'frechet_distance': 46.0233, 'frechet_audio_distance': 7.3721, 'kl_sigmoid': 7.0712, 'kl_softmax': 2.1406
    e9, gs9 (5245): 'frechet_distance': 46.1362, 'frechet_audio_distance': 7.3185, 'kl_sigmoid': 7.1951, 'kl_softmax': 2.1417

