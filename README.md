
# Attempt 5: Try to train the model without music attend to motion

This is to check the quality of the data. Remove the third line of the attention map

We start with e12 from 3.2

    e3, gs3: 'frechet_distance': 46.579, 'frechet_audio_distance': 7.1813, 'kl_sigmoid': 7.0455, 'kl_softmax': 2.125
    e6, gs3: 'frechet_distance': 48.0502, 'frechet_audio_distance': 7.5015, 'kl_sigmoid': 7.2512, 'kl_softmax': 2.1142
    e9, gs3: 'frechet_distance': 47.9048, 'frechet_audio_distance': 7.3032, 'kl_sigmoid': 7.5377, 'kl_softmax': 2.2265
    e12, gs3: 'frechet_distance': 45.5392, 'frechet_audio_distance': 7.7305, 'kl_sigmoid': 7.28, 'kl_softmax': 2.1356


## 5.2: Use 100% music weight to finetune
Also, I removed the motion-to-music attention. Now this should be absolutely independent.


## 5.3: Try to train on spotify dataset
first we clean the dataset using preprocessing/clean_spotify.py.
Then we start from e12 3.2
