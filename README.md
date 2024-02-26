# Ablation 2: Use an independent VQVAE
Branch from 22, and reference to  12. We use the code in data/motion/aligned_motion_code_122
to train the model. First, we try not initializing the codebook. Second, we try initializing the codebook
with original music codebook.

## Ablation 2.1:
I marked line 110-115 in load_mm_lm_model to remove the motion embedding initialization