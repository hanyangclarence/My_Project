
# Attempt 20: Use Joint Music Motion Generation for trainable self-attn
Branch from attempt17, adopt similar changes as attempt19.

    e30
    BLEU Score: 0.29189615016177334
    BLEU-4 Score: 0.17069683688800666
    METEOR Score: 0.3077157489565554
    ROUGE Score: 0.38570979560604896
    BERT Score: 0.8949564099311829

    {'Matching_score': tensor(4.1324), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3683), 'R_precision_top_2': tensor(0.5582), 'R_precision_top_3': tensor(0.6682), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.4778753480402656, 'precisions': [0.4778753480402656], 'brevity_penalty': 1.0, 'length_ratio': 1.361939210081092, 'translation_length': 46690, 'reference_length': 34282, 'time_elapsed': 0.3529033660888672}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3655), 'CIDEr': tensor(0.0521), 
    'Bleu_4': {'score': 0.06873947788585938, 'precisions': [0.4778753480402656, 0.17367157716383488, 0.03323933364666562, 0.00809336695794968], 'brevity_penalty': 1.0, 'length_ratio': 1.361939210081092, 'translation_length': 46690, 'reference_length': 34282, 'time_elapsed': 0.974848747253418}, 
    'Bert_F1': tensor(0.3737)}



    e42
    BLEU Score: 0.2773431601814606
    BLEU-4 Score: 0.156095714359197
    METEOR Score: 0.28082390508227467
    ROUGE Score: 0.36312774248664553
    BERT Score: 0.8884144425392151

    {'Matching_score': tensor(4.0638), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3860), 'R_precision_top_2': tensor(0.5778), 'R_precision_top_3': tensor(0.6877), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.4789639870285032, 'precisions': [0.4789639870285032], 'brevity_penalty': 1.0, 'length_ratio': 1.3672481185461758, 'translation_length': 46872, 'reference_length': 34282, 'time_elapsed': 0.33312010765075684}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3684), 'CIDEr': tensor(0.0536), 
    'Bleu_4': {'score': 0.0712245570578487, 'precisions': [0.4789639870285032, 0.1760239947511482, 0.033757796257796256, 0.009042118772605297], 'brevity_penalty': 1.0, 'length_ratio': 1.3672481185461758, 'translation_length': 46872, 'reference_length': 34282, 'time_elapsed': 0.9216864109039307}, 
    'Bert_F1': tensor(0.3776)}
