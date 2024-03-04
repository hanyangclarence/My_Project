# Ablation 5: Rerun full model
Branch from 22, remove some randomness

    e12
    KL_Sigmoid:  6.34453; KL:  1.91774; PSNR: nan SSIM: nan ISc:  1.81842 (0.037843); KID: nan (nan) FD: 39.17185; FAD: 8.04711

    e15
    KL_Sigmoid:  6.32038; KL:  1.93402; PSNR: nan SSIM: nan ISc:  1.70860 (0.028257); KID: nan (nan) FD: 36.96839; FAD: 5.93645
    gs4: 'fid_k': 16.70991652472722, 'fid_m': 13.137362038803474, 'div_k': 7.0280965889722875, 'div_m': 7.139979033058171, 0.21660525585535462 0.2276(for pre-detected beats)
    gs3: 'fid_k': 11.712561702391255, 'fid_m': (13.436980449795243-8.875712563557055e-09j), 'div_k': 6.902553649382913, 'div_m': 6.967174832302591
    gs4, seed32: 'fid_k': 12.808291530636922, 'fid_m': (13.485018558936915-4.6405308673904604e-08j), 'div_k': 8.612601591647003, 'div_m': 8.29996225855688,  0.2283269589015701 0.22272614360668894(for pre-detected beats)
    gs4, seed37: 'fid_k': 16.70914929875532, 'fid_m': 8.850796353049986, 'div_k': 6.739906287093293, 'div_m': 7.387597926621852, 0.22957364989376466 0.21709814202304425(for pre-detected beats)

    on aist new
    default: 'fid_k': 15.183961102535378, 'fid_m': 22.232131796520036, 'div_k': 8.237611915387813, 'div_m': 8.003370746669251
    seed11: 'fid_k': 15.407142354126464, 'fid_m': 32.544456510507004, 'div_k': 9.793763423332779, 'div_m': 8.819269811251747
    seed21: 'fid_k': 13.152855589132074, 'fid_m': 27.287778588611353, 'div_k': 9.312562778657606, 'div_m': 8.585447642880512
    seed31: 'fid_k': 15.598019587350393, 'fid_m': 30.700125149491548, 'div_k': 9.981161872365321, 'div_m': 8.807336584105292
    seed41: 'fid_k': 12.700208912673446, 'fid_m': 26.734862588084212, 'div_k': 8.36429745959015, 'div_m': 8.421603645716038

    e18
    KL_Sigmoid:  6.32082; KL:  1.93005; PSNR: nan SSIM: nan ISc:  1.62990 (0.015407); KID: nan (nan) FD: 40.18120; FAD: 8.31243

    e21
    KL_Sigmoid:  6.60899; KL:  1.97002; PSNR: nan SSIM: nan ISc:  1.66043 (0.022882); KID: nan (nan) FD: 39.97653; FAD: 7.84137

# Ablation 5.2: Train with vocal

    e09
    'frechet_distance': 44.0925, 'frechet_audio_distance': 6.6353, 'kl_sigmoid': 6.9534, 'kl_softmax': 2.0419

    e12
    'frechet_distance': 42.9537, 'frechet_audio_distance': 4.2991, 'kl_sigmoid': 7.564, 'kl_softmax': 2.1225 CLAP: 0.2896
    KL_Sigmoid:  6.52679; KL:  1.95412; PSNR: nan SSIM: nan ISc:  1.77271 (0.025851); KID: nan (nan) FD: 31.62746; FAD: 4.11130

    e15
    'frechet_distance': 39.8829, 'frechet_audio_distance': 4.9938, 'kl_sigmoid': 7.4121, 'kl_softmax': 2.1162

    e18
    'frechet_distance': 41.3883, 'frechet_audio_distance': 6.4271, 'kl_sigmoid': 6.9695, 'kl_softmax': 2.0753