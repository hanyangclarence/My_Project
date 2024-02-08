
# Attempt 7: Remove motion codebook
Maybe the shared codebook really works?


## 7.1: Start from 3.2, and only remove motion codebook.
Give 100% weight to motion. I didn't add the two modality embeddings

    e12 looks the worse. the figure doesn't move! e18 and e24 looks the best. 
    e6, gs3: 'fid_k': 129.48658286654975, 'fid_m': (22.23535111738674-1.929959971357385e-08j), 'div_k': 13.23966866168526, 'div_m': 6.301560356837485
    e12, gs3: 'fid_k': 298.2807435118218, 'fid_m': (17.57926128873752-1.1546669408114234e-08j), 'div_k': 15.868399671534432, 'div_m': 6.8915849746878335
    e18, gs3: 'fid_k': 12.340159639706243, 'fid_m': (7.417536668340411-7.865986450148827e-09j), 'div_k': 6.353150162067568, 'div_m': 6.451497082792662
    e24, gs3: 'fid_k': 8.23244474093886, 'fid_m': (13.967190187684949-1.5383677137762822e-08j), 'div_k': 7.862186014514038, 'div_m': 7.439229777040024

Then we continue with e18

    musiccap
    e6, gs4: 'frechet_distance': 44.9365, 'frechet_audio_distance': 7.2114, 'kl_sigmoid': 7.0391, 'kl_softmax': 2.1157
    e12, gs4: 'frechet_distance': 44.8817, 'frechet_audio_distance': 7.3861, 'kl_sigmoid': 7.1694, 'kl_softmax': 2.1071
    e18, gs4: 'frechet_distance': 45.9183, 'frechet_audio_distance': 6.9739, 'kl_sigmoid': 7.0027, 'kl_softmax': 2.1141
    e24, gs4: 'frechet_distance': 45.1197, 'frechet_audio_distance': 7.3052, 'kl_sigmoid': 7.0751, 'kl_softmax': 2.138