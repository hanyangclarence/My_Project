
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

    e42:
    BLEU Score: 0.2753548040400152
    BLEU-4 Score: 0.1668940012331658
    METEOR Score: 0.3143511208332761
    ROUGE Score: 0.3727499614237411
    BERT Score: 0.891626238822937

    {'Matching_score': tensor(4.7318), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3075), 'R_precision_top_2': tensor(0.4869), 'R_precision_top_3': tensor(0.5983), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.5032317636195752, 'precisions': [0.5032317636195752], 'brevity_penalty': 1.0, 'length_ratio': 1.2004550492970072, 'translation_length': 41154, 'reference_length': 34282, 'time_elapsed': 0.3323345184326172}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3774), 'CIDEr': tensor(0.0443), 
    'Bleu_4': {'score': 0.07223089554141846, 'precisions': [0.5032317636195752, 0.1948969100059527, 0.03388071546303644, 0.008191556395715185], 'brevity_penalty': 1.0, 'length_ratio': 1.2004550492970072, 'translation_length': 41154, 'reference_length': 34282, 'time_elapsed': 0.6049282550811768}, 
    'Bert_F1': tensor(0.3855)}


    e54:
    BLEU Score: 0.2682202810695649
    BLEU-4 Score: 0.16090434334279216
    METEOR Score: 0.3173363882370643
    ROUGE Score: 0.36785435200552896
    BERT Score: 0.89219069480896

    {'Matching_score': tensor(4.5808), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3266), 'R_precision_top_2': tensor(0.5045), 'R_precision_top_3': tensor(0.6238), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.48346620450606587, 'precisions': [0.48346620450606587], 'brevity_penalty': 1.0, 'length_ratio': 1.2623242517939444, 'translation_length': 43275, 'reference_length': 34282, 'time_elapsed': 0.3124423027038574}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3704), 'CIDEr': tensor(0.0430), 
    'Bleu_4': {'score': 0.06830525150313371, 'precisions': [0.48346620450606587, 0.18112029478748176, 0.030759969039360145, 0.008081598070844331], 'brevity_penalty': 1.0, 'length_ratio': 1.2623242517939444, 'translation_length': 43275, 'reference_length': 34282, 'time_elapsed': 0.873852014541626}, 
    'Bert_F1': tensor(0.3733)}

