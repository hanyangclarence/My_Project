
# Attempt 16: Use original causal attention mask
Branch from 15, and use causal attention mask, but still no cross attention

    e30:
    BLEU Score: 0.2561438141933511
    BLEU-4 Score: 0.1649320164976517
    METEOR Score: 0.33086423707545537
    ROUGE Score: 0.3742569373771276
    BERT Score: 0.8920141458511353

    'Matching_score': tensor(8.1166), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0658), 'R_precision_top_2': tensor(0.1188), 'R_precision_top_3': tensor(0.1615), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.2583867258091014, 'precisions': [0.2583867258091014], 'brevity_penalty': 1.0, 'length_ratio': 1.7755673531299223, 'translation_length': 60870, 'reference_length': 34282, 'time_elapsed': 0.339292049407959}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.2140), 'CIDEr': tensor(0.0100), 
    'Bleu_4': {'score': 0.015827601571305756, 'precisions': [0.2583867258091014, 0.06105092282175248, 0.004573345020770609, 0.0008698893997763141], 'brevity_penalty': 1.0, 'length_ratio': 1.7755673531299223, 'translation_length': 60870, 'reference_length': 34282, 'time_elapsed': 0.6483256816864014}, 
    'Bert_F1': tensor(0.1438)}

    e42:
    BLEU Score: 0.2803995969650303
    BLEU-4 Score: 0.16405368494122585
    METEOR Score: 0.3334606319481774
    ROUGE Score: 0.38452373643693466
    BERT Score: 0.8948946595191956

    {'Matching_score': tensor(7.7638), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0835), 'R_precision_top_2': tensor(0.1453), 'R_precision_top_3': tensor(0.1951), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3464036974718324, 'precisions': [0.34640369747183236], 'brevity_penalty': 1.0, 'length_ratio': 1.369552534857943, 'translation_length': 46951, 'reference_length': 34282, 'time_elapsed': 0.318148136138916}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.2543), 'CIDEr': tensor(0.0156), 
    'Bleu_4': {'score': 0.02707864688417431, 'precisions': [0.34640369747183236, 0.10157876271781079, 0.009906895925724215, 0.0015423566044873848], 'brevity_penalty': 1.0, 'length_ratio': 1.369552534857943, 'translation_length': 46951, 'reference_length': 34282, 'time_elapsed': 0.8887276649475098}, 
    'Bert_F1': tensor(0.2014)}
