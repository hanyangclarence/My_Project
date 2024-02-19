
# Attempt 19: Use Joint Text Generation
Branch from attempt16

    e30:
    BLEU Score: 0.2656880001116898
    BLEU-4 Score: 0.16151577517632404
    METEOR Score: 0.3218372617412865
    ROUGE Score: 0.37781474357529
    BERT Score: 0.890350341796875

    {'Matching_score': tensor(8.2591), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0651), 'R_precision_top_2': tensor(0.1119), 'R_precision_top_3': tensor(0.1546), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.42239040055764393, 'precisions': [0.42239040055764393], 'brevity_penalty': 1.0, 'length_ratio': 1.1717227699667463, 'translation_length': 40169, 'reference_length': 34282, 'time_elapsed': 0.41909193992614746}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.2935), 'CIDEr': tensor(0.0150), 
    'Bleu_4': {'score': 0.032596927868815975, 'precisions': [0.42239040055764393, 0.15522753175993106, 0.0169619536142493, 0.0010151916174177876], 'brevity_penalty': 1.0, 'length_ratio': 1.1717227699667463, 'translation_length': 40169, 'reference_length': 34282, 'time_elapsed': 0.6093325614929199}, 
    'Bert_F1': tensor(0.2516)}