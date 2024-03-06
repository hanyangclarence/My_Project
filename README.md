
# Ablation 1: Multi Task Training

Branch from 22. Train the model with text-to-music-motion, music-to-motion, motion-to-music jointly. Each task has 1/3 prob
to be chosen

    e09
    musiccap: KL_Sigmoid: -1.00000; KL: -1.00000; PSNR: nan SSIM: nan ISc:  1.74317 (0.035071); KID: nan (nan) FD: 39.32741; FAD: 6.41020 LSD: nan
    'fid_k': 71.77948297178517, 'fid_m': (37.32811213471576-4.461348073736861e-09j), 'div_k': 10.07154671148313, 'div_m': 6.768311946878465

    e12
    musiccap: KL_Sigmoid:  6.53567; KL:  1.96374; PSNR: nan SSIM: nan ISc:  1.66183 (0.019639); KID: nan (nan) FD: 40.36085; FAD: 6.88864 LSD: nan
    'fid_k': 39.849298058884415, 'fid_m': 45.40374917105626, 'div_k': 10.776742870027539, 'div_m': 6.377351811862444

    e15
    musiccap: KL_Sigmoid:  6.74169; KL:  1.96009; PSNR: nan SSIM: nan ISc:  1.62045 (0.013945); KID: nan (nan) FD: 42.62187; FAD: 8.18120 LSD: nan
    'fid_k': 46.74992737991923, 'fid_m': 32.50009591179025, 'div_k': 8.453274003281907, 'div_m': 6.060379975825515    

    e18
    musiccap: KL_Sigmoid:  6.39716; KL:  1.94688; PSNR: nan SSIM: nan ISc:  1.66234 (0.022927); KID: nan (nan) FD: 37.34306; FAD: 6.16977 LSD: nan
    'fid_k': 34.91336174020023, 'fid_m': (38.92935922833519-2.468175531062223e-08j), 'div_k': 9.042028725692129, 'div_m': 5.923471393066904
    'frechet_distance': 45.9957, 'frechet_audio_distance': 6.9443, 'kl_sigmoid': 6.7534, 'kl_softmax': 2.0676  CLAP: 0.2632
