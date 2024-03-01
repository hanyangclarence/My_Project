# Ablation 2: Use an independent VQVAE
Branch from 22, and reference to  12. We use the code in data/motion/aligned_motion_code_122
to train the model. First, we try not initializing the codebook. Second, we try initializing the codebook
with original music codebook.

## Ablation 2.1:
I marked line 110-115 in load_mm_lm_model to remove the motion embedding initialization

    e09
    musiccap: KL_Sigmoid:  6.48029; KL:  1.91514; PSNR: nan SSIM: nan ISc:  1.67678 (0.036570); KID: nan (nan) FD: 40.14126; FAD: 7.61303

    e18
    musiccap: KL_Sigmoid:  6.41853; KL:  1.96502; PSNR: nan SSIM: nan ISc:  1.65440 (0.033718); KID: nan (nan) FD: 38.75110; FAD: 7.27269




# Ablation 2.2
Use motion codebook with initialization, but independent motion code
