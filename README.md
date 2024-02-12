
# Attempt 12: Use new motion VQVAE and code

Branch from attempt 3.2. Use 1.0 motion weight, and only change
the aligned motion code and motion vqvae

Stage 1:

    e09, gs3: 'fid_k': 235.6203330951148, 'fid_m': (9.86371430376012-1.552785222694555e-08j), 'div_k': 15.1685111994641, 'div_m': 7.7087843295071306 (in many case, the figure doesn't move)
    e12, gs3: 'fid_k': 3060.2708956465804, 'fid_m': 22.46572980373041, 'div_k': 36.90096796889668, 'div_m': 8.995490414729485, 'div_k_gt': 9.250034450753242 (this looks not good... Even more not-moving figures)
    e15, gs3: 'fid_k': 7102.518614613557, 'fid_m': (20.83734084586642-2.4365315330330254e-08j), 'div_k': 38.699554989704716, 'div_m': 7.820394845104537
    e18, gs3: 'fid_k': 560.2740539752267, 'fid_m': (16.64585190851414-9.657329557887005e-09j), 'div_k': 23.420776782675066, 'div_m': 7.214375260120783 (seems the best among these, but still not good)

Stage 2: 
