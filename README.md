
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
        gs3:
        gs2: 'fid_k': 224.19747269334593, 'fid_m': 20.06257223459795, 'div_k': 15.438705773490625, 'div_m': 6.413277538134018  (This actualy looks pretty good)
    e12, gs4: 'fid_k': 112.45380101693789, 'fid_m': 15.528898173937591, 'div_k': 13.324163940776982, 'div_m': 8.33782428788667  (seems better than above, but not much)
        gs3:
        gs2: 'fid_k': 220.93180770348653, 'fid_m': (19.45247030570843-3.952053609513531e-08j), 'div_k': 16.62285497243195, 'div_m': 8.957078768819434
    e15, gs4: 'fid_k': 427.22886171660286, 'fid_m': (11.597547834143-2.4312307453493375e-08j), 'div_k': 15.219845264680277, 'div_m': 8.068017122197976 (e15 and e18 looks the best)
        gs3:
        gs2: 'fid_k': 131.74858201774967, 'fid_m': 8.41858595892679, 'div_k': 12.25577055722122, 'div_m': 7.712340715985201
    e18, gs4: 'fid_k': 19.271818441624433, 'fid_m': 15.26410536163538, 'div_k': 8.606044639592186, 'div_m': 6.682529765929665
        gs3:
        gs2: 'fid_k': 61.29407676700146, 'fid_m': 13.53510840005417, 'div_k': 10.330810923782014, 'div_m': 6.933205456191481

Then, we start with epoch 12 for stage 2


### Attempt 3.2.2: Start from epoch 18 in previous training




### Attempt 3.3: Start from epoch 18, and then use new dataset
