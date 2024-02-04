
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

Then, we start with epoch 21 for stage 2

    musiccap:
    e3, gs4: 'frechet_distance': 47.4936, 'frechet_audio_distance': 6.8062, 'kl_sigmoid': 7.9617, 'kl_softmax': 2.2738
    e9, gs4: 'frechet_distance': 54.4776, 'frechet_audio_distance': 10.7176, 'kl_sigmoid': 8.0889, 'kl_softmax': 2.2975
    e15, gs4: 'frechet_distance': 50.9853, 'frechet_audio_distance': 7.7529, 'kl_sigmoid': 8.2627, 'kl_softmax': 2.2335

    motion 5s:
    e3, gs4: 'fid_k': 70.70517068905119, 'fid_m': (153.66033973700593-4.612126388054071e-09j), 'div_k': 12.522745895165132, 'div_m': 9.603234493530447
    e6, gs4: 'fid_k': 11.156277326846379, 'fid_m': 17.826827186118685, 'div_k': 7.3277379339700826, 'div_m': 5.853206861126782
    e9, gs4: 'fid_k': 15.95070830150516, 'fid_m': (82.10346131551319-1.7139885308149898e-08j), 'div_k': 9.671764290061683, 'div_m': 9.264130287486706
    e12, gs4: 'fid_k': 12.649011079664149, 'fid_m': (89.09835066359285-1.2198238397178547e-08j), 'div_k': 9.173138292319532, 'div_m': 10.116231299671973
    e15, gs4: 'fid_k': 19.910169531154366, 'fid_m': (50.619042407358634-1.4068795960988061e-08j), 'div_k': 10.58246388030823, 'div_m': 8.933915988620175  (don't know why but this one really looks the best, but also it has severe overfit)


### Attempt 3.2.2: Start from epoch 18 in previous training

    musiccap:
    e6, gs4:
    e9, gs4:
    e15, gs4:

    motion 5s:
    e3, gs4: 'fid_k': 10.686002712654258, 'fid_m': (68.5818790888735-1.7550071969744246e-08j), 'div_k': 7.571955755067909, 'div_m': 8.849621651058346
    e6, gs4: 'fid_k': 21.957979374958796, 'fid_m': 39.89605115364397, 'div_k': 9.470350434410665, 'div_m': 7.351177875915897
    e9, gs4: 'fid_k': 34.52649500572767, 'fid_m': 52.403497285147566, 'div_k': 11.50238682024455, 'div_m': 7.887636879843347
    e12, gs4: 'fid_k': 8.544377506794874, 'fid_m': 50.89859033000302, 'div_k': 9.042598850308453, 'div_m': 8.868599993640364
    e15, gs4: 'fid_k': 19.03689911357084, 'fid_m': 63.01254745541796, 'div_k': 10.205156568377845, 'div_m': 9.085922721103415


### Attempt 3.3: Start from epoch 18, and then use new dataset
