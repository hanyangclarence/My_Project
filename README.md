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
    Still use the (4, 2048) size codebook, but it's independent one
    Total loss: total loss: 0.08064708113670349

Then we use its code to train the model in attempt 1.1


## Attempt 1.3: Train a new vqvae of smaller size

