
# Attempt 10: Only differs from main in conditioning
Branch from attempt9, and remove mixture of expert.

    musiccaps
    e12, gs4: 'frechet_distance': 43.6644, 'frechet_audio_distance': 6.4895, 'kl_sigmoid': 7.1309, 'kl_softmax': 2.0328
    e18, gs4: 'frechet_distance': 46.8459, 'frechet_audio_distance': 7.2716, 'kl_sigmoid': 6.9755, 'kl_softmax': 2.0816
    e24, gs4: 'frechet_distance': 47.4334, 'frechet_audio_distance': 6.8211, 'kl_sigmoid': 6.8692, 'kl_softmax': 2.0927
    e30, gs4: 'frechet_distance': 44.8467, 'frechet_audio_distance': 6.9024, 'kl_sigmoid': 7.119, 'kl_softmax': 2.1225

    motion 5s (These seems not bad actually. But also not that good)
    e12, gs3: 'fid_k': 41.40899748706633, 'fid_m': 14.89629808770566, 'div_k': 9.558381355887937, 'div_m': 6.3849409423730314 (some jitters, but seems diverse)
    e18, gs3: 'fid_k': 14.091032341182114, 'fid_m': (38.37919121282718-3.480385290487854e-08j), 'div_k': 7.083083274210974, 'div_m': 7.427184782647502
    e24, gs3: 'fid_k': 17.75501232362558, 'fid_m': (12.716440440621511-1.546089827428189e-08j), 'div_k': 6.324777237339695, 'div_m': 7.2403181823998395


Train with seed 42
    
    e12, gs4: 'frechet_distance': 44.8129, 'frechet_audio_distance': 7.0727, 'kl_sigmoid': 6.8485, 'kl_softmax': 2.0718
    e15, gs4: 'frechet_distance': 46.4674, 'frechet_audio_distance': 8.0166, 'kl_sigmoid': 6.9369, 'kl_softmax': 2.1033
    e18, gs4: 'frechet_distance': 45.7732, 'frechet_audio_distance': 7.3562, 'kl_sigmoid': 6.9483, 'kl_softmax': 2.0863
    e24, gs4: 