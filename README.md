
# Attempt 22: Use new dataloader in train caption
Branch from 17, and add a new data loader that does not use aligned motion. Instead, we use raw motion.

But first, we extract motion features with extract_motion_code.py

The motion code is extracted by zero-padding the remaining parts

    e30

    {'Matching_score': tensor(4.7150), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3438), 'R_precision_top_2': tensor(0.5327), 'R_precision_top_3': tensor(0.6510), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.5256635256635257, 'precisions': [0.5256635256635257], 'brevity_penalty': 1.0, 'length_ratio': 1.0803628726445365, 'translation_length': 37037, 'reference_length': 34282, 'time_elapsed': 0.6451377868652344}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3941), 'CIDEr': tensor(0.0463), 
    'Bleu_4': {'score': 0.09058338469903418, 'precisions': [0.5256635256635257, 0.22404920678420268, 0.04395182405306336, 0.013006666939343122], 'brevity_penalty': 1.0, 'length_ratio': 1.0803628726445365, 'translation_length': 37037, 'reference_length': 34282, 'time_elapsed': 1.0791034698486328}, 
    'Bert_F1': tensor(0.4018)}   

    e42

    {'Matching_score': tensor(4.3245), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3869), 'R_precision_top_2': tensor(0.5708), 'R_precision_top_3': tensor(0.6861), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.5046300540937013, 'precisions': [0.5046300540937013], 'brevity_penalty': 1.0, 'length_ratio': 1.2726212006300683, 'translation_length': 43628, 'reference_length': 34282, 'time_elapsed': 0.37638020515441895}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 'ROUGE_L': tensor(0.4025), 'CIDEr': tensor(0.0592), 
    'Bleu_4': {'score': 0.0900961397710664, 'precisions': [0.5046300540937013, 0.20691316697098802, 0.04401748212055852, 0.014336340206185566], 'brevity_penalty': 1.0, 'length_ratio': 1.2726212006300683, 'translation_length': 43628, 'reference_length': 34282, 'time_elapsed': 0.9403698444366455}, 
    'Bert_F1': tensor(0.4110)}

    e54

    (completed)
    {'Matching_score': tensor(4.0418), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3899), 'R_precision_top_2': tensor(0.5864), 'R_precision_top_3': tensor(0.7047), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': {'score': 0.5290103973264018, 'precisions': [0.5290103973264018], 'brevity_penalty': 1.0, 'length_ratio': 1.0539086195088543, 'translation_length': 43088, 'reference_length': 40884, 'time_elapsed': 0.3450002670288086}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 'ROUGE_L': tensor(0.3950), 'CIDEr': tensor(0.0620), 
    'Bleu_4': {'score': 0.09803511639095609, 'precisions': [0.5290103973264018, 0.221959213401311, 0.049644970414201184, 0.015845795033612293], 'brevity_penalty': 1.0, 'length_ratio': 1.0539086195088543, 'translation_length': 43088, 'reference_length': 40884, 'time_elapsed': 0.9336991310119629}, 
    'Bert_F1': tensor(0.4041)}

    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3469 (2): 0.1752 (3): 0.0929 (4): 0.0489
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2787
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0620
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.4041
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 3.1769 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9873 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.4851 CInt: 0.0000;(top 2) Mean: 0.6905 CInt: 0.0000;(top 3) Mean: 0.7933 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5162 CInt: 0.0000;(top 2) Mean: 0.7056 CInt: 0.0000;(top 3) Mean: 0.8026 CInt: 0.0000;





    tm2t gt results
    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.5806 (2): 0.4540 (3): 0.3807 (4): 0.3393
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.5082
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 1.9213
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.4169
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 2.9992 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9718 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.4991 CInt: 0.0000;(top 2) Mean: 0.6942 CInt: 0.0000;(top 3) Mean: 0.7940 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5121 CInt: 0.0000;(top 2) Mean: 0.7114 CInt: 0.0000;(top 3) Mean: 0.7959 CInt: 0.0000;

    motiongpt gt results
    {'Matching_score': tensor(4.0418), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3899), 'R_precision_top_2': tensor(0.5864), 'R_precision_top_3': tensor(0.7047), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': {'score': 0.5803204047217538, 'precisions': [0.5803204047217538], 'brevity_penalty': 1.0, 'length_ratio': 1.3030103274005713, 'translation_length': 59300, 'reference_length': 45510, 'time_elapsed': 0.28790712356567383}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.4992), 
    'CIDEr': tensor(1.9171), 
    'Bleu_4': {'score': 0.33877902443286156, 'precisions': [0.5803204047217538, 0.35450819672131145, 0.2671358873870271, 0.23968435901957327], 'brevity_penalty': 1.0, 'length_ratio': 1.3030103274005713, 'translation_length': 59300, 'reference_length': 45510, 'time_elapsed': 0.8038599491119385}, 
    'Bert_F1': tensor(0.4166)}



# Attempt 22.2: Use repeated motion code
