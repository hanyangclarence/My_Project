
# Attempt 15: Start working with text
We start from only changing to fully attention mask in encoder!
    
    e30:
    BLEU Score: 0.2519156533175739
    BLEU-4 Score: 0.15802908212164032
    METEOR Score: 0.3257725033854701
    ROUGE Score: 0.3622084466255796
    BERT Score: 0.8938219547271729

    {'Matching_score': tensor(6.3040), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.1672), 'R_precision_top_2': tensor(0.2829), 'R_precision_top_3': tensor(0.3760), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3854676561924312, 'precisions': [0.3854676561924312], 'brevity_penalty': 1.0, 'length_ratio': 1.1083950761332477, 'translation_length': 37998, 'reference_length': 34282, 'time_elapsed': 0.3515961170196533}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.2846), 'CIDEr': tensor(0.0227), 
    'Bleu_4': {'score': 0.03924585067270877, 'precisions': [0.3854676561924312, 0.12126501390450269, 0.01401742889954739, 0.0036206218024399843], 'brevity_penalty': 1.0, 'length_ratio': 1.1083950761332477, 'translation_length': 37998, 'reference_length': 34282, 'time_elapsed': 0.8554337024688721}, 
    'Bert_F1': tensor(0.2658)}

    (My previous sota)
    {'Matching_score': tensor(3.8882), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3705), 'R_precision_top_2': tensor(0.5656), 'R_precision_top_3': tensor(0.6858), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.43248412636199735, 'precisions': [0.43248412636199735], 'brevity_penalty': 1.0, 'length_ratio': 1.8605973980514556, 'translation_length': 63785, 'reference_length': 34282, 'time_elapsed': 0.369067907333374}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3296), 'CIDEr': tensor(0.0484), 
    'Bleu_4': {'score': 0.056870417065092806, 'precisions': [0.43248412636199735, 0.1353437715014516, 0.026140487065152637, 0.006836338066683594], 'brevity_penalty': 1.0, 'length_ratio': 1.8605973980514556, 'translation_length': 63785, 'reference_length': 34282, 'time_elapsed': 1.026357889175415}, 
    'Bert_F1': tensor(0.2590)}

    e42:
    BLEU Score: 0.27188509858023896
    BLEU-4 Score: 0.1627285587785718
    METEOR Score: 0.3354281253561334
    ROUGE Score: 0.3812189555871715
    BERT Score: 0.8964552879333496

    {'Matching_score': tensor(8.0127), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0451), 'R_precision_top_2': tensor(0.0859), 'R_precision_top_3': tensor(0.1279), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.29169950425829416, 'precisions': [0.29169950425829416], 'brevity_penalty': 1.0, 'length_ratio': 1.1473951344729012, 'translation_length': 39335, 'reference_length': 34282, 'time_elapsed': 0.33183884620666504}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.1888), 'CIDEr': tensor(0.0096), 
    'Bleu_4': {'score': 0.015935150347416573, 'precisions': [0.29169950425829416, 0.05523776999914625, 0.004653718126878454, 0.0008599095225632781], 'brevity_penalty': 1.0, 'length_ratio': 1.1473951344729012, 'translation_length': 39335, 'reference_length': 34282, 'time_elapsed': 0.6454355716705322}, 
    'Bert_F1': tensor(0.1149)}

    e54:
    BLEU Score: 0.2665478783590597
    BLEU-4 Score: 0.16420802537506513
    METEOR Score: 0.3305566184802281
    ROUGE Score: 0.376438816174244
    BERT Score: 0.8956520557403564

    e66:
    BLEU Score: 0.27179322073797907
    BLEU-4 Score: 0.158808726119993
    METEOR Score: 0.3518165103648842
    ROUGE Score: 0.3848171960642398
    BERT Score: 0.8971327543258667

# Attempt 15.2: Not using caption from mullama

    e30:
    BLEU Score: 0.139395308616235
    BLEU-4 Score: 0.04742057990664781
    METEOR Score: 0.176557184747175
    ROUGE Score: 0.2257973034393317
    BERT Score: 0.8679806590080261

    e42:
    BLEU Score: 0.13728777938217024
    BLEU-4 Score: 0.047076029084690225
    METEOR Score: 0.16681262528182988
    ROUGE Score: 0.22839277596104776
    BERT Score: 0.8672478795051575

# Attempt 15.3: Use only Humanml3d as motion training data
    Oh noo.. The motion is totally unrelated

    e30

    'Matching_score': tensor(8.7954), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0594), 'R_precision_top_2': tensor(0.1147), 'R_precision_top_3': tensor(0.1653), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.3893933708567855, 'precisions': [0.3893933708567855], 'brevity_penalty': 1.0, 'length_ratio': 1.166063823580888, 'translation_length': 39975, 'reference_length': 34282, 'time_elapsed': 0.33843278884887695}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3013), 'CIDEr': tensor(0.0118), 
    'Bleu_4': {'score': 0.035525522491923964, 'precisions': [0.3893933708567855, 0.14217837278850723, 0.01250672830320109, 0.0023003614853762734], 'brevity_penalty': 1.0, 'length_ratio': 1.166063823580888, 'translation_length': 39975, 'reference_length': 34282, 'time_elapsed': 0.9317972660064697}, 
    'Bert_F1': tensor(0.2784)

    
    e45

    'Matching_score': tensor(8.2764), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0670), 'R_precision_top_2': tensor(0.1248), 'R_precision_top_3': tensor(0.1799), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.33549166753696275, 'precisions': [0.3354916675369627], 'brevity_penalty': 1.0, 'length_ratio': 1.9551659763141007, 'translation_length': 67027, 'reference_length': 34282, 'time_elapsed': 0.36396288871765137}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.2570), 'CIDEr': tensor(0.0172), 
    'Bleu_4': {'score': 0.024834110687101136, 'precisions': [0.3354916675369627, 0.08796613136827362, 0.008254455529973566, 0.0015613806278587043], 'brevity_penalty': 1.0, 'length_ratio': 1.9551659763141007, 'translation_length': 67027, 'reference_length': 34282, 'time_elapsed': 0.7070560455322266}, 
    'Bert_F1': tensor(0.1427)


    e60

    'Matching_score': tensor(7.8787), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.0694), 'R_precision_top_2': tensor(0.1286), 'R_precision_top_3': tensor(0.1858), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.34762781724107694, 'precisions': [0.3476278172410769], 'brevity_penalty': 1.0, 'length_ratio': 1.6631176710810338, 'translation_length': 57015, 'reference_length': 34282, 'time_elapsed': 0.32509589195251465}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.2818), 'CIDEr': tensor(0.0157), 
    'Bleu_4': {'score': 0.027277588309260043, 'precisions': [0.3476278172410769, 0.10628751017626233, 0.009789605742138495, 0.0015306007607986134], 'brevity_penalty': 1.0, 'length_ratio': 1.6631176710810338, 'translation_length': 57015, 'reference_length': 34282, 'time_elapsed': 0.9048268795013428}, 
    'Bert_F1': tensor(0.2397)}
