# Ablation 4: Removing MoE only
Branch from 22, and try only removing MoE, and still keep the separate codebooks.

    e09
    musiccap: KL_Sigmoid:  6.29416; KL:  1.84242; PSNR: nan SSIM: nan ISc:  1.72206 (0.034916); KID: nan (nan) FD: 37.57727; FAD: 6.49722 
    'fid_k': 17.187739653382522, 'fid_m': 14.314392467449956, 'div_k': 6.554643239087219, 'div_m': 6.233596009894491        

    e12
    KL_Sigmoid:  6.48866; KL:  1.91772; PSNR: nan SSIM: nan ISc:  1.74677 (0.037820); KID: nan (nan) FD: 36.94533; FAD: 6.46155 LSD: nan
    'fid_k': 20.362322652844924, 'fid_m': (13.277542602202885-1.601861929207889e-08j), 'div_k': 5.845069222874865, 'div_m': 6.7512375264143865

    e15
    KL_Sigmoid:  6.55686; KL:  1.96194; PSNR: nan SSIM: nan ISc:  1.62914 (0.028783); KID: nan (nan) FD: 39.98008; FAD: 6.81090 LSD: nan
    'fid_k': 11.156709696502915, 'fid_m': 19.132127340538347, 'div_k': 8.519514864713386, 'div_m': 7.390804981108892

    e18
    KL_Sigmoid:  6.63622; KL:  1.97591; PSNR: nan SSIM: nan ISc:  1.57819 (0.025281); KID: nan (nan) FD: 40.69068; FAD: 7.03615 LSD: nan
    'fid_k': 13.535094013208123, 'fid_m': 14.691361635423938, 'div_k': 7.663482721960266, 'div_m': 7.431759897966183


