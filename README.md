!!! Ideas!

1, According to BEiT-3, we can adopt Mixture of Expert. Add another feed-forward layer for motion tokens.
The parallel generation is still this.

2, According to BLIP, we can train another fully self-attention layer in music-motion transformer decoder
during training captioning. Just replace the previous causal attention layer.

3, According to VLMo, we can train the model in two stage: (1) Motion only training. Train motion only with
predict-next-token, and freeze most of the model, except for motion codebook, motion feed-forward, motion classification
head, ... (2) Music and motion joint training with all the module trainable.



# Attempt 1: Use separate music and motion code.

## Attempt 1.1: Directly use two codebook in music motion lm. 
The motion codebook is initialized from music codebook. Use the previous motion vqvae with a shared codebook
    
    music: 

    motion: run test_music2motion_5s.py, measure the score on 300 results
    e18, gs 4: 'fid_k': 27.185054473942486, 'fid_m': 16.3414047318757, 'div_k': 4.9015415329774354, 'div_m': 5.064991518975368
    e24, gs 4: 'fid_k': 51.134297124903675, 'fid_m': (16.822706175418986-1.0577419096926252e-08j), 'div_k': 3.3964420876398664, 'div_m': 5.540540530280259
    e30, gs 4: 'fid_k': 17.4427346668961, 'fid_m': 12.256695386086477, 'div_k': 6.873758819829195, 'div_m': 6.986007423868679


## Attempt 1.2: Train a new vqvae that have trainable codebook.

### Attempt 1.2.1
    The structure of the trainable codebook is still the same. It is still of shape (4, 2048)
    The test loss is not good: total loss: 0.1118762344121933
    I guess this is because when training, waveform is provided. But when testing, the input is zero waveform.

### Attempt 1.2.2 Use independent motion vqvae
    Still use the (4, 2048) size codebook, but it's independent one. Code in unimumo/models/motion_vqvae_v2.py
    Total loss: total loss: 0.08064708113670349
    Weight stored as pretrained/motion_vqvae_122.ckpt

Then we use its code to train the model in attempt 1.1


## Attempt 1.3: Train a new vqvae of smaller size


## Attempt 1.4: Use separate feed forward, like BEiT 3: 
The idea of Mixture of Expert!
    
    musiccap
    e15, gs4: 'frechet_distance': 45.1839, 'frechet_audio_distance': 6.5972, 'kl_softmax': 2.1922
    e18, gs4: 'frechet_distance': 46.5567, 'frechet_audio_distance': 7.5426, 'kl_softmax': 2.1534
    e24, gs4: 'frechet_distance': 47.8238, 'frechet_audio_distance': 7.8338, 'kl_softmax': 2.1283
    e30, gs4: 'frechet_distance': 46.51, 'frechet_audio_distance': 7.8986, 'kl_softmax': 2.1483

    motion: 5s  These all looks not bad
    e15, gs4: 'fid_k': 28.16595031103958, 'fid_m': 25.128384154562852, 'div_k': 7.916131069933956, 'div_m': 7.241416175136864  (some still have jittering)
    e18, gs4: 'fid_k': 45.79209097850983, 'fid_m': (41.70675454020242-1.2339236570220533e-08j), 'div_k': 10.092344975242913, 'div_m': 7.847883006686484
    e24, gs4: 'fid_k': 15.166639026381716, 'fid_m': (21.320750258078064-1.1161775287076629e-08j), 'div_k': 9.847920319428678, 'div_m': 7.43546487721844
    e30, gs4: 'fid_k': 2634.200991246921, 'fid_m': 11.220858005511474, 'div_k': 17.992683423963545, 'div_m': 7.095945158517746  (can observe obvious overfitting)

### Attempt 1.4.2: Use new music4all_ignore file
Ohh nooooo! All of them are really noisy! The music part!!

    musiccap
    e21, gs4: 'frechet_distance': 51.8921, 'frechet_audio_distance': 9.7393, 'kl_softmax': 2.3049 
    e27, gs4: 'frechet_distance': 49.7816, 'frechet_audio_distance': 6.9168, 'kl_softmax': 2.1997
    e33, gs4: 'frechet_distance': 46.01, 'frechet_audio_distance': 5.8568, 'kl_softmax': 2.1717
    e39, gs4: 'frechet_distance': 50.6006, 'frechet_audio_distance': 6.646, 'kl_softmax': 2.2048

    motion 5s
    e21, gs4: 'fid_k': 63.378289547145755, 'fid_m': (27.93885819957844-2.0869182025679013e-08j), 'div_k': 9.794431733433884, 'div_m': 7.316785662735054
    e27, gs4: 'fid_k': 27.712417059538723, 'fid_m': (13.57193132866231-2.9372374329782815e-08j), 'div_k': 8.398052284959434, 'div_m': 6.680871987664447
    e33, gs4: 'fid_k': 21.319195019659333, 'fid_m': 19.0015063008562, 'div_k': 9.97412735908858, 'div_m': 6.847419910808338
    e39, gs4: 'fid_k': 152.46546993133998, 'fid_m': (11.90965979896363-1.7476277353810287e-08j), 'div_k': 12.743685098356227, 'div_m': 7.044677717685699

