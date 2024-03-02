# Ablation 2: Use an independent VQVAE
Branch from 22, and reference to  12. We use the code in data/motion/aligned_motion_code_122
to train the model. First, we try not initializing the codebook. Second, we try initializing the codebook
with original music codebook.

## Ablation 2.1:
I marked line 110-115 in load_mm_lm_model to remove the motion embedding initialization

    e09
    musiccap: KL_Sigmoid:  6.48029; KL:  1.91514; PSNR: nan SSIM: nan ISc:  1.67678 (0.036570); KID: nan (nan) FD: 40.14126; FAD: 7.61303
    'fid_k': 9.944894486744289, 'fid_m': 33.72649981803606, 'div_k': 8.194376741855042, 'div_m': 7.535954384149915

    e12
    musiccap: KL_Sigmoid:  6.52585; KL:  2.01038; PSNR: nan SSIM: nan ISc:  1.63621 (0.031905); KID: nan (nan) FD: 41.24742; FAD: 6.29425 LSD: nan
    'fid_k': 13.552077023740054, 'fid_m': (19.52993503865393-2.5271796908558268e-08j), 'div_k': 7.258716442519539, 'div_m': 8.04408471705257

    e15
    musiccap: KL_Sigmoid:  6.54431; KL:  1.96355; PSNR: nan SSIM: nan ISc:  1.58956 (0.019734); KID: nan (nan) FD: 40.92968; FAD: 7.80536
    'fid_k': 36.195112996728795, 'fid_m': (20.00425573751994-2.8274888493620048e-08j), 'div_k': 4.5745831682894345, 'div_m': 7.295213613171242

    e18
    musiccap: KL_Sigmoid:  6.41853; KL:  1.96502; PSNR: nan SSIM: nan ISc:  1.65440 (0.033718); KID: nan (nan) FD: 38.75110; FAD: 7.27269
    'fid_k': 12.627846222430136, 'fid_m': (16.098959310971225-1.6683107971641813e-08j), 'div_k': 6.898497220630796, 'div_m': 8.317730375646615




# Ablation 2.2
Use motion codebook with initialization, but independent motion code. That is, I remove the mark in line 110-115 in load_mm_lm_model

    e09
    KL_Sigmoid:  6.97517; KL:  2.07408; PSNR: nan SSIM: nan ISc:  1.54937 (0.024867); KID: nan (nan) FD: 47.85132; FAD: 10.31581

    e12
