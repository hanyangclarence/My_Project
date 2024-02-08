
# Attempt 5: Try to train the model without music attend to motion

This is to check the quality of the data. Remove the third line of the attention map

We start with e12 from 3.2

    e3, gs3: 'frechet_distance': 46.579, 'frechet_audio_distance': 7.1813, 'kl_sigmoid': 7.0455, 'kl_softmax': 2.125
    e6, gs3: 'frechet_distance': 48.0502, 'frechet_audio_distance': 7.5015, 'kl_sigmoid': 7.2512, 'kl_softmax': 2.1142
    e9, gs3: 'frechet_distance': 47.9048, 'frechet_audio_distance': 7.3032, 'kl_sigmoid': 7.5377, 'kl_softmax': 2.2265
    e12, gs3: 'frechet_distance': 45.5392, 'frechet_audio_distance': 7.7305, 'kl_sigmoid': 7.28, 'kl_softmax': 2.1356


## 5.2: Use 100% music weight to finetune
Also, I removed the motion-to-music attention. Now this should be absolutely independent.

    e3, gs4: 'frechet_distance': 46.5273, 'frechet_audio_distance': 6.9686, 'kl_sigmoid': 6.8755, 'kl_softmax': 2.0913
    e6, gs4: 'frechet_distance': 46.036, 'frechet_audio_distance': 7.3363, 'kl_sigmoid': 7.1793, 'kl_softmax': 2.0906
    e9, gs4: 'frechet_distance': 46.0216, 'frechet_audio_distance': 7.797, 'kl_sigmoid': 7.0673, 'kl_softmax': 2.0948
    e12, gs4: 'frechet_distance': 45.3989, 'frechet_audio_distance': 8.6292, 'kl_sigmoid': 7.2042, 'kl_softmax': 2.104


## 5.3: Try to train on spotify dataset
first we clean the dataset using preprocessing/clean_spotify.py.
Then we start from e12 3.2

    e6, gs3: 'frechet_distance': 50.943, 'frechet_audio_distance': 7.9871, 'kl_sigmoid': 7.4582, 'kl_softmax': 2.2509
    e12, gs3: 'frechet_distance': 49.3025, 'frechet_audio_distance': 7.7787, 'kl_sigmoid': 7.6986, 'kl_softmax': 2.2866
    e18, gs3: 'frechet_distance': 52.3687, 'frechet_audio_distance': 10.1335, 'kl_sigmoid': 7.6917, 'kl_softmax': 2.2882
    e24, gs3: 'frechet_distance': 50.4968, 'frechet_audio_distance': 8.1698, 'kl_sigmoid': 7.9336, 'kl_softmax': 2.301
