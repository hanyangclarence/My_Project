
# Attempt 17: Use trainable self-attn

    e30
    BLEU Score: 0.27536695397255606
    BLEU-4 Score: 0.14732411657907374
    METEOR Score: 0.27439814742868346
    ROUGE Score: 0.37227896243683634
    BERT Score: 0.8901135921478271

    On original data
    {'Matching_score': tensor(5.9848), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2166), 'R_precision_top_2': tensor(0.3406), 'R_precision_top_3': tensor(0.4227), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3089829789811618, 'precisions': [0.3089829789811618], 'brevity_penalty': 1.0, 'length_ratio': 1.9262586780234525, 'translation_length': 66036, 'reference_length': 34282, 'time_elapsed': 0.3977963924407959}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3120), 'CIDEr': tensor(0.0359), 
    'Bleu_4': {'score': 0.03636377308038973, 'precisions': [0.3089829789811618, 0.09964424320827943, 0.016497527972937808, 0.003442469597754911], 'brevity_penalty': 1.0, 'length_ratio': 1.9262586780234525, 'translation_length': 66036, 'reference_length': 34282, 'time_elapsed': 0.6968119144439697}, 
    'Bert_F1': tensor(0.2883)}

    On aligned data
    {'Matching_score': tensor(4.1293), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3719), 'R_precision_top_2': tensor(0.5663), 'R_precision_top_3': tensor(0.6837), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.4765049307939562, 'precisions': [0.4765049307939562], 'brevity_penalty': 1.0, 'length_ratio': 1.3340236858993058, 'translation_length': 45733, 'reference_length': 34282, 'time_elapsed': 0.33948373794555664}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3640), 'CIDEr': tensor(0.0557), 
    'Bleu_4': {'score': 0.07557576032166154, 'precisions': [0.4765049307939562, 0.1733635072345138, 0.038161806057684586, 0.010348468848996832], 'brevity_penalty': 1.0, 'length_ratio': 1.3340236858993058, 'translation_length': 45733, 'reference_length': 34282, 'time_elapsed': 0.9013702869415283}, 
    'Bert_F1': tensor(0.3773)}

    On repeat data
    {'Matching_score': tensor(4.4691), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3342), 'R_precision_top_2': tensor(0.5050), 'R_precision_top_3': tensor(0.6157), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.4668114211269135, 'precisions': [0.4668114211269135], 'brevity_penalty': 1.0, 'length_ratio': 1.3434163701067616, 'translation_length': 46055, 'reference_length': 34282, 'time_elapsed': 0.34845709800720215}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3536), 'CIDEr': tensor(0.0499), 
    'Bleu_4': {'score': 0.07094531514104953, 'precisions': [0.4668114211269135, 0.1666786115291813, 0.03470249316305127, 0.009382376669555085], 'brevity_penalty': 1.0, 'length_ratio': 1.3434163701067616, 'translation_length': 46055, 'reference_length': 34282, 'time_elapsed': 0.8835737705230713}, 
    'Bert_F1': tensor(0.3664)}


    e42

    BLEU Score: 0.28399657736931067
    BLEU-4 Score: 0.1649144429266015
    METEOR Score: 0.2973607269166294
    ROUGE Score: 0.38103665177478113
    BERT Score: 0.8927560448646545
    
    On original data
    {'Matching_score': tensor(6.1102), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2195), 'R_precision_top_2': tensor(0.3483), 'R_precision_top_3': tensor(0.4342),  
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3758665277827664, 'precisions': [0.3758665277827664], 'brevity_penalty': 1.0, 'length_ratio': 1.6242342920483053, 'translation_length': 55682, 'reference_length': 34282, 'time_elapsed': 0.33853650093078613}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3205), 'CIDEr': tensor(0.0344), 
    'Bleu_4': {'score': 0.044123304264845696, 'precisions': [0.3758665277827664, 0.12296546634036437, 0.01850285472615775, 0.004432171531999814], 'brevity_penalty': 1.0, 'length_ratio': 1.6242342920483053, 'translation_length': 55682, 'reference_length': 34282, 'time_elapsed': 0.9408025741577148}, 
    'Bert_F1': tensor(0.3070)}

    On aligned data
    {'Matching_score': tensor(4.0901), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3898), 'R_precision_top_2': tensor(0.5742), 'R_precision_top_3': tensor(0.6901), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.46468042190192044, 'precisions': [0.46468042190192044], 'brevity_penalty': 1.0, 'length_ratio': 1.3883087334461233, 'translation_length': 47594, 'reference_length': 34282, 'time_elapsed': 0.33879923820495605}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3616), 'CIDEr': tensor(0.0535), 
    'Bleu_4': {'score': 0.06911904642020918, 'precisions': [0.46468042190192044, 0.1639706898935435, 0.03328911790214785, 0.008998457407301605], 'brevity_penalty': 1.0, 'length_ratio': 1.3883087334461233, 'translation_length': 47594, 'reference_length': 34282, 'time_elapsed': 0.8871791362762451}, 
    'Bert_F1': tensor(0.3717)}

    On repeat data
    {'Matching_score': tensor(4.4688), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3368), 'R_precision_top_2': tensor(0.5076), 'R_precision_top_3': tensor(0.6267), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.4609607392249109, 'precisions': [0.4609607392249109], 'brevity_penalty': 1.0, 'length_ratio': 1.3826789568869962, 'translation_length': 47401, 'reference_length': 34282, 'time_elapsed': 0.31923651695251465}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3563), 'CIDEr': tensor(0.0505), 
    'Bleu_4': {'score': 0.06709736923446732, 'precisions': [0.4609607392249109, 0.16206457585927556, 0.03117229357327796, 0.008703645190015223], 'brevity_penalty': 1.0, 'length_ratio': 1.3826789568869962, 'translation_length': 47401, 'reference_length': 34282, 'time_elapsed': 0.8823819160461426}, 
    'Bert_F1': tensor(0.3631)}

    e54

    BLEU Score: 0.27602032536093757
    BLEU-4 Score: 0.15324382588439978
    METEOR Score: 0.279770358244704
    ROUGE Score: 0.3763577191381629
    BERT Score: 0.8909918665885925

    On original data
    {'Matching_score': tensor(6.1558), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2283), 'R_precision_top_2': tensor(0.3578), 'R_precision_top_3': tensor(0.4373), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.40328855023059956, 'precisions': [0.40328855023059956], 'brevity_penalty': 1.0, 'length_ratio': 1.454699259086401, 'translation_length': 49870, 'reference_length': 34282, 'time_elapsed': 0.31893491744995117}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3229), 'CIDEr': tensor(0.0371), 
    'Bleu_4': {'score': 0.049432233137233154, 'precisions': [0.40328855023059956, 0.13309541533476377, 0.02205988716910169, 0.005042647926613379], 'brevity_penalty': 1.0, 'length_ratio': 1.454699259086401, 'translation_length': 49870, 'reference_length': 34282, 'time_elapsed': 0.8841416835784912}, 
    'Bert_F1': tensor(0.3121)}

    On aligned data
    {'Matching_score': tensor(4.0909), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3833), 'R_precision_top_2': tensor(0.5756), 'R_precision_top_3': tensor(0.6875), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.4734075043630017, 'precisions': [0.4734075043630017], 'brevity_penalty': 1.0, 'length_ratio': 1.3371448573595472, 'translation_length': 45840, 'reference_length': 34282, 'time_elapsed': 0.33861255645751953}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3615), 'CIDEr': tensor(0.0587), 
    'Bleu_4': {'score': 0.07252336335785257, 'precisions': [0.4734075043630017, 0.17003649985592162, 0.03548921170690023, 0.009683628052447973], 'brevity_penalty': 1.0, 'length_ratio': 1.3371448573595472, 'translation_length': 45840, 'reference_length': 34282, 'time_elapsed': 0.8972592353820801}, 
    'Bert_F1': tensor(0.3746)}
