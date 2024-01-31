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

### Attempt 1.4.2: Use new music4all_ignore file

