
# Attempt 19: Use Joint Text Generation
Branch from attempt16

    e30:
    BLEU Score: 0.2656880001116898
    BLEU-4 Score: 0.16151577517632404
    METEOR Score: 0.3218372617412865
    ROUGE Score: 0.37781474357529
    BERT Score: 0.890350341796875

    {'Matching_score': tensor(5.6648), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.2273), 'R_precision_top_2': tensor(0.3633), 'R_precision_top_3': tensor(0.4614), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.5023221833742108, 'precisions': [0.5023221833742108], 'brevity_penalty': 1.0, 'length_ratio': 1.117962779301091, 'translation_length': 38326, 'reference_length': 34282, 'time_elapsed': 0.3134739398956299}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3661), 'CIDEr': tensor(0.0290), 
    'Bleu_4': {'score': 0.05797373587185538, 'precisions': [0.5023221833742108, 0.19457954878406095, 0.028061735818801364, 0.0041184241199782425], 'brevity_penalty': 1.0, 'length_ratio': 1.117962779301091, 'translation_length': 38326, 'reference_length': 34282, 'time_elapsed': 0.6049795150756836}, 
    'Bert_F1': tensor(0.3605)}