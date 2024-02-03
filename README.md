
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

    motion 5s:
    e9, gs4: 'fid_k': 619.7093621404829, 'fid_m': 35.55905909780583, 'div_k': 20.729237431796765, 'div_m': 6.619866691496859
    e15, gs4: 'fid_k': 16.96088608297036, 'fid_m': 54.771122981836854, 'div_k': 7.6212745574804455, 'div_m': 8.22999427852822
    e18, gs4: 'fid_k': 53.626457909767424, 'fid_m': 84.52090147339291, 'div_k': 11.009721285273532, 'div_m': 7.751532624185152
    e21, gs4: 'fid_k': 26.913671192995054, 'fid_m': 66.43637617688299, 'div_k': 10.31575540674437, 'div_m': 8.03104855731711
    e24, gs4: 'fid_k': 106.40655109919402, 'fid_m': (149.00719319868173-7.219853910922441e-09j), 'div_k': 13.924959823925759, 'div_m': 8.524272498316323  (don't know why but this one is that bad, there is a lot of jittering)
    e27, gs4: 'fid_k': 36.07839712098317, 'fid_m': (59.4669769213578-1.9886433470787522e-08j), 'div_k': 11.67602386889839, 'div_m': 7.451998793266188  (This seems already overfit)

