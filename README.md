
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
    e9, gs4: 'fid_k': 68.83738869885056, 'fid_m': (17.684981490009946-9.741114776681073e-09j), 'div_k': 10.617871300202681, 'div_m': 6.1446957984629815  (but the figures are not moving. Actually not good)
        gs3: 'fid_k': 128.24142254175936, 'fid_m': 21.103918784141406, 'div_k': 12.805416163658492, 'div_m': 6.269958510707894
        gs2: 'fid_k': 224.19747269334593, 'fid_m': 20.06257223459795, 'div_k': 15.438705773490625, 'div_m': 6.413277538134018  (This actualy looks pretty good)
    e12, gs4: 'fid_k': 112.45380101693789, 'fid_m': 15.528898173937591, 'div_k': 13.324163940776982, 'div_m': 8.33782428788667  (seems better than above, but not much)
        gs3: 'fid_k': 2712.3282812815132, 'fid_m': 19.647375036713896, 'div_k': 21.680088419754988, 'div_m': 8.35134476570012
        gs2: 'fid_k': 220.93180770348653, 'fid_m': (19.45247030570843-3.952053609513531e-08j), 'div_k': 16.62285497243195, 'div_m': 8.957078768819434
    e15, gs4: 'fid_k': 427.22886171660286, 'fid_m': (11.597547834143-2.4312307453493375e-08j), 'div_k': 15.219845264680277, 'div_m': 8.068017122197976 (e15 and e18 looks the best)
        gs3: 'fid_k': 92.8946713328439, 'fid_m': 8.850249473934952, 'div_k': 11.378143821058014, 'div_m': 7.714392996075406
        gs2: 'fid_k': 131.74858201774967, 'fid_m': 8.41858595892679, 'div_k': 12.25577055722122, 'div_m': 7.712340715985201
    e18, gs4: 'fid_k': 19.271818441624433, 'fid_m': 15.26410536163538, 'div_k': 8.606044639592186, 'div_m': 6.682529765929665
        gs3: 'fid_k': 27.524691587516514, 'fid_m': (14.672116079105834-1.4957636271182823e-08j), 'div_k': 9.788371401116201, 'div_m': 6.653282440524702
        gs2: 'fid_k': 61.29407676700146, 'fid_m': 13.53510840005417, 'div_k': 10.330810923782014, 'div_m': 6.933205456191481

Then, we start with epoch 12 for stage 2

    musiccap:
    e3, gs3:
    e6, gs3:
    e9, gs3:
    e12, gs3:
    e15, gs3:



### Attempt 3.2.2: Start from epoch 18 in previous training

    musiccap:
    e3, gs3: 'frechet_distance': 47.7047, 'frechet_audio_distance': 7.6948, 'kl_sigmoid': 7.0535, 'kl_softmax': 2.1388
    e6, gs3: 'frechet_distance': 47.9272, 'frechet_audio_distance': 7.2428, 'kl_sigmoid': 7.3651, 'kl_softmax': 2.1897
    e9, gs3: 'frechet_distance': 45.8944, 'frechet_audio_distance': 7.8981, 'kl_sigmoid': 7.2816, 'kl_softmax': 2.1132
    e12, gs3 (4201): 'frechet_distance': 46.1751, 'frechet_audio_distance': 7.9808, 'kl_sigmoid': 7.1604, 'kl_softmax': 2.1214
    e15, gs3 (2762): 'frechet_distance': 47.4896, 'frechet_audio_distance': 8.2619, 'kl_sigmoid': 7.5606, 'kl_softmax': 2.1893




## Attempt 3.3: Start from epoch 18, and then use new dataset

## Attempt 3.4: Give lower weight to motion loss
Seems that motion is too separated from music, so that training any fine-tuning cannot train joint generation 
together.

Change motion weight to 0.5 

After stage 1:

    e9 and e12 looks not good. Motions are too fast and unatural. But actually e15 looks not bad. e21 is also not bad, but might be overfit
    music2motion 5s:
    e09, gs3: 'fid_k': 2288.562317682805, 'fid_m': 40.457945425164894, 'div_k': 29.67374393470876, 'div_m': 8.693808652321032
    e12, gs3: 'fid_k': 113.6714743678225, 'fid_m': (43.706101933106666-1.3099851403703215e-08j), 'div_k': 12.742204713911782, 'div_m': 8.424984482262312
    e15, gs3: 'fid_k': 261.77090647682616, 'fid_m': (14.43330952082637-4.0122607344579195e-08j), 'div_k': 15.295245946076934, 'div_m': 7.24095122361861
    e21, gs3: 'fid_k': 521.3320762026016, 'fid_m': 15.126177963665889, 'div_k': 16.28407096623112, 'div_m': 7.48125927775467

Then start with e12

    This sounds better though
    e3, gs4: 'frechet_distance': 46.6195, 'frechet_audio_distance': 7.1361, 'kl_sigmoid': 6.7574, 'kl_softmax': 2.0593
    e6, gs4:
    e9, gs4:
    e12, gs4: 'frechet_distance': 46.4299, 'frechet_audio_distance': 7.2615, 'kl_sigmoid': 7.107, 'kl_softmax': 2.0907


