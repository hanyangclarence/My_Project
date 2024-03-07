
# Ablation 1: Multi Task Training

Branch from 22. Train the model with text-to-music-motion, music-to-motion, motion-to-music jointly. Each task has 1/3 prob
to be chosen

    e09
    musiccap: KL_Sigmoid: -1.00000; KL: -1.00000; PSNR: nan SSIM: nan ISc:  1.74317 (0.035071); KID: nan (nan) FD: 39.32741; FAD: 6.41020 LSD: nan
    'fid_k': 71.77948297178517, 'fid_m': (37.32811213471576-4.461348073736861e-09j), 'div_k': 10.07154671148313, 'div_m': 6.768311946878465

    e12
    musiccap: KL_Sigmoid:  6.53567; KL:  1.96374; PSNR: nan SSIM: nan ISc:  1.66183 (0.019639); KID: nan (nan) FD: 40.36085; FAD: 6.88864 LSD: nan
    'fid_k': 39.849298058884415, 'fid_m': 45.40374917105626, 'div_k': 10.776742870027539, 'div_m': 6.377351811862444
    'frechet_distance': 43.7212, 'frechet_audio_distance': 6.6404, 'kl_sigmoid': 7.1058, 'kl_softmax': 2.0555  CLAP: 0.2511
    seed21: 'fid_k': 67.40042788484385, 'fid_m': 75.55374414812646, 'div_k': 10.614179639934564, 'div_m': 8.714783860007788 beat align: 0.19437556928716881 
    seed31: 'fid_k': 120.50073922025013, 'fid_m': 73.89096688032339, 'div_k': 12.839079416138405, 'div_m': 8.342638062978933 beat align: 0.21573360538296407

    e15
    musiccap: KL_Sigmoid:  6.74169; KL:  1.96009; PSNR: nan SSIM: nan ISc:  1.62045 (0.013945); KID: nan (nan) FD: 42.62187; FAD: 8.18120 LSD: nan
    'fid_k': 46.74992737991923, 'fid_m': 32.50009591179025, 'div_k': 8.453274003281907, 'div_m': 6.060379975825515    
    'frechet_distance': 45.4507, 'frechet_audio_distance': 7.5612, 'kl_sigmoid': 7.0688, 'kl_softmax': 2.0345  CLAP: 0.2462

    e18
    musiccap: KL_Sigmoid:  6.39716; KL:  1.94688; PSNR: nan SSIM: nan ISc:  1.66234 (0.022927); KID: nan (nan) FD: 37.34306; FAD: 6.16977 LSD: nan
    'frechet_distance': 45.9957, 'frechet_audio_distance': 6.9443, 'kl_sigmoid': 6.7534, 'kl_softmax': 2.0676  CLAP: 0.2632   
    beat alignment: 0.18800827177614504
