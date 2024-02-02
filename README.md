
# Attempt 3: Add pretraining

## Attempt 3.1: Pretrain music-to-motion generation
Use the task music-to-motion generation, with the attention mask from attempt 2. Freeze most of the parameters,
allowing only motion codebooks, motion feedforward and motion classification head.

In this attempt, we also use cross-attention mask for music-to-motion.
Stage 1, train only music2motion:

    motion 5s:
    e15, gs4: 'fid_k': 361.222391800925, 'fid_m': 172.91844110083684, 'div_k': 18.503923313595966, 'div_m': 9.07695181746479
    e21, gs4: 'fid_k': 22.718215363046795, 'fid_m': (134.89619792251452-1.758622421406383e-08j), 'div_k': 8.034932193503272, 'div_m': 7.822581291708688
    e27, gs4: 'fid_k': 14.849613856472757, 'fid_m': 74.06959053703045, 'div_k': 7.005912038739919, 'div_m': 7.79773544779396
    (The first two have a lot of jitters, the last one is better but far from good. Also have jitters, and motion moves very fast and unnatural)


## Attempt 3.2: Give full weight to motion loss
Change motion weight to 1.0, and lr to 1e-5

