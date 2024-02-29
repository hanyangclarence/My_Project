
# Ablation 1: Multi Task Training

Branch from 22. Train the model with text-to-music-motion, music-to-motion, motion-to-music jointly. Each task has 1/3 prob
to be chosen

    e09
    musiccap: KL_Sigmoid: -1.00000; KL: -1.00000; PSNR: nan SSIM: nan ISc:  1.74317 (0.035071); KID: nan (nan) FD: 39.32741; FAD: 6.41020 LSD: nan

    e12
    musiccap: KL_Sigmoid:  6.53567; KL:  1.96374; PSNR: nan SSIM: nan ISc:  1.66183 (0.019639); KID: nan (nan) FD: 40.36085; FAD: 6.88864 LSD: nan

    e15
    musiccap: KL_Sigmoid:  6.74169; KL:  1.96009; PSNR: nan SSIM: nan ISc:  1.62045 (0.013945); KID: nan (nan) FD: 42.62187; FAD: 8.18120 LSD: nan

    e18
    musiccap: KL_Sigmoid:  6.39716; KL:  1.94688; PSNR: nan SSIM: nan ISc:  1.66234 (0.022927); KID: nan (nan) FD: 37.34306; FAD: 6.16977 LSD: nan
