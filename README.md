
# Attempt 13: Rerun attempt 1 with separate conditioning
Both MoE and separate codebook are applied. Train directly from scratch.
In attempt1, there are still some bugs in conditioning, but still gets
a good results. Now I run again with a fixed conditioning.

Also, I improved the conditioning part. In training, I removed the conditioning
choices of music2motion and motion2music, and use the built-in cfg dropout to 
achieve this.

