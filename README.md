
# Attempt 17: Use trainable self-attn

    e30

    On original data
    {'Matching_score': tensor(5.9848), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2166), 'R_precision_top_2': tensor(0.3406), 'R_precision_top_3': tensor(0.4227), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3089829789811618, 'precisions': [0.3089829789811618], 'brevity_penalty': 1.0, 'length_ratio': 1.9262586780234525, 'translation_length': 66036, 'reference_length': 34282, 'time_elapsed': 0.3977963924407959}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3120), 'CIDEr': tensor(0.0359), 
    'Bleu_4': {'score': 0.03636377308038973, 'precisions': [0.3089829789811618, 0.09964424320827943, 0.016497527972937808, 0.003442469597754911], 'brevity_penalty': 1.0, 'length_ratio': 1.9262586780234525, 'translation_length': 66036, 'reference_length': 34282, 'time_elapsed': 0.6968119144439697}, 
    'Bert_F1': tensor(0.2883)}


    e42


    
    On original data
    {'Matching_score': tensor(6.1102), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2195), 'R_precision_top_2': tensor(0.3483), 'R_precision_top_3': tensor(0.4342),  
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3758665277827664, 'precisions': [0.3758665277827664], 'brevity_penalty': 1.0, 'length_ratio': 1.6242342920483053, 'translation_length': 55682, 'reference_length': 34282, 'time_elapsed': 0.33853650093078613}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3205), 'CIDEr': tensor(0.0344), 
    'Bleu_4': {'score': 0.044123304264845696, 'precisions': [0.3758665277827664, 0.12296546634036437, 0.01850285472615775, 0.004432171531999814], 'brevity_penalty': 1.0, 'length_ratio': 1.6242342920483053, 'translation_length': 55682, 'reference_length': 34282, 'time_elapsed': 0.9408025741577148}, 
    'Bert_F1': tensor(0.3070)}

    e54


    On original data
    {'Matching_score': tensor(6.1558), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2283), 'R_precision_top_2': tensor(0.3578), 'R_precision_top_3': tensor(0.4373), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.40328855023059956, 'precisions': [0.40328855023059956], 'brevity_penalty': 1.0, 'length_ratio': 1.454699259086401, 'translation_length': 49870, 'reference_length': 34282, 'time_elapsed': 0.31893491744995117}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3229), 'CIDEr': tensor(0.0371), 
    'Bleu_4': {'score': 0.049432233137233154, 'precisions': [0.40328855023059956, 0.13309541533476377, 0.02205988716910169, 0.005042647926613379], 'brevity_penalty': 1.0, 'length_ratio': 1.454699259086401, 'translation_length': 49870, 'reference_length': 34282, 'time_elapsed': 0.8841416835784912}, 
    'Bert_F1': tensor(0.3121)}
