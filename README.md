
# Attempt 16: Use original causal attention mask
Branch from 15, and use causal attention mask, but still no cross attention

    e30:
    BLEU Score: 0.2561438141933511
    BLEU-4 Score: 0.1649320164976517
    METEOR Score: 0.33086423707545537
    ROUGE Score: 0.3742569373771276
    BERT Score: 0.8920141458511353

    {'Matching_score': tensor(4.9418), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2686), 'R_precision_top_2': tensor(0.4361), 'R_precision_top_3': tensor(0.5491), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.35328608344793194, 'precisions': [0.35328608344793194], 'brevity_penalty': 1.0, 'length_ratio': 1.930254944285631, 'translation_length': 66173, 'reference_length': 34282, 'time_elapsed': 0.3316459655761719}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3018), 'CIDEr': tensor(0.0328), 
    'Bleu_4': {'score': 0.03622897474357229, 'precisions': [0.35328608344793194, 0.10266711844716589, 0.013395406794621069, 0.0035457684053373144], 'brevity_penalty': 1.0, 'length_ratio': 1.930254944285631, 'translation_length': 66173, 'reference_length': 34282, 'time_elapsed': 0.665956974029541}, 
    'Bert_F1': tensor(0.2327)}

    e42:
    BLEU Score: 0.2803995969650303
    BLEU-4 Score: 0.16405368494122585
    METEOR Score: 0.3334606319481774
    ROUGE Score: 0.38452373643693466
    BERT Score: 0.8948946595191956

    {'Matching_score': tensor(4.7133), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3073), 'R_precision_top_2': tensor(0.4735), 'R_precision_top_3': tensor(0.5885), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.45194327205808815, 'precisions': [0.45194327205808815], 'brevity_penalty': 1.0, 'length_ratio': 1.4582871477743422, 'translation_length': 49993, 'reference_length': 34282, 'time_elapsed': 0.31902432441711426}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3482), 'CIDEr': tensor(0.0430), 
    'Bleu_4': {'score': 0.059289814780356165, 'precisions': [0.45194327205808815, 0.15352534008777868, 0.02697050551669431, 0.006603395268012298], 'brevity_penalty': 1.0, 'length_ratio': 1.4582871477743422, 'translation_length': 49993, 'reference_length': 34282, 'time_elapsed': 0.8883171081542969}, 
    'Bert_F1': tensor(0.3144)}
