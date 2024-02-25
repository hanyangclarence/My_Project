
# Attempt 22: Use new dataloader in train caption
Branch from 17, and add a new data loader that does not use aligned motion. Instead, we use raw motion.

But first, we extract motion features with extract_motion_code.py

The motion code is extracted by zero-padding the remaining parts

    e30

    BLEU Score: 0.2502841084795902
    BLEU-4 Score: 0.16351874827177476
    METEOR Score: 0.34317886321420504
    ROUGE Score: 0.37312134018578674
    BERT Score: 0.8938829898834229

    {'Matching_score': tensor(4.7150), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3438), 'R_precision_top_2': tensor(0.5327), 'R_precision_top_3': tensor(0.6510), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.5256635256635257, 'precisions': [0.5256635256635257], 'brevity_penalty': 1.0, 'length_ratio': 1.0803628726445365, 'translation_length': 37037, 'reference_length': 34282, 'time_elapsed': 0.6451377868652344}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3941), 'CIDEr': tensor(0.0463), 
    'Bleu_4': {'score': 0.09058338469903418, 'precisions': [0.5256635256635257, 0.22404920678420268, 0.04395182405306336, 0.013006666939343122], 'brevity_penalty': 1.0, 'length_ratio': 1.0803628726445365, 'translation_length': 37037, 'reference_length': 34282, 'time_elapsed': 1.0791034698486328}, 
    'Bert_F1': tensor(0.4018)}   

    e42

    BLEU Score: 0.2728034459482026
    BLEU-4 Score: 0.1610319296285683
    METEOR Score: 0.3355172024000024
    ROUGE Score: 0.3817655313749064
    BERT Score: 0.8977652192115784

    {'Matching_score': tensor(4.3245), 'gt_Matching_score': tensor(3.6037), 
    'R_precision_top_1': tensor(0.3869), 'R_precision_top_2': tensor(0.5708), 'R_precision_top_3': tensor(0.6861), 
    'gt_R_precision_top_1': tensor(0.4079), 'gt_R_precision_top_2': tensor(0.6078), 'gt_R_precision_top_3': tensor(0.7195), 
    'Bleu_1': {'score': 0.5046300540937013, 'precisions': [0.5046300540937013], 'brevity_penalty': 1.0, 'length_ratio': 1.2726212006300683, 'translation_length': 43628, 'reference_length': 34282, 'time_elapsed': 0.37638020515441895}, 
    'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 'ROUGE_L': tensor(0.4025), 'CIDEr': tensor(0.0592), 
    'Bleu_4': {'score': 0.0900961397710664, 'precisions': [0.5046300540937013, 0.20691316697098802, 0.04401748212055852, 0.014336340206185566], 'brevity_penalty': 1.0, 'length_ratio': 1.2726212006300683, 'translation_length': 43628, 'reference_length': 34282, 'time_elapsed': 0.9403698444366455}, 
    'Bert_F1': tensor(0.4110)}

    e54

    BLEU Score: 0.26398237249421697
    BLEU-4 Score: 0.16055160540851926
    METEOR Score: 0.31197643831627603
    ROUGE Score: 0.3692726558458997
    BERT Score: 0.8913907408714294

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

    e66
    BLEU Score: 0.26096739866440294
    BLEU-4 Score: 0.14625695605381528
    METEOR Score: 0.29057097911264346
    ROUGE Score: 0.36931510966571246
    BERT Score: 0.8917101621627808

    {'Matching_score': tensor(3.9815), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.4082), 'R_precision_top_2': tensor(0.6149), 'R_precision_top_3': tensor(0.7248), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': tensor(0.5288), 'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.4014), 'CIDEr': tensor(0.0622), 
    'Bleu_4': tensor(0.0928), 
    'Bert_F1': tensor(0.4091)}

    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3536 (2): 0.1735 (3): 0.0889 (4): 0.0447
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2834
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0622
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.4089
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 3.0592 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9649 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.5213 CInt: 0.0000;(top 2) Mean: 0.7091 CInt: 0.0000;(top 3) Mean: 0.8022 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5125 CInt: 0.0000;(top 2) Mean: 0.7026 CInt: 0.0000;(top 3) Mean: 0.7991 CInt: 0.0000;





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

    tm2t test motion gpt:
    {'Bleu_1': 0.3999965309697605, 'Bleu_2': 0.23229768870638673, 'Bleu_3': 0.10790567814838915, 'Bleu_4': 0.050031062564891114, 
    'ROUGE_L': 0.3366613968243719, 'CIDEr': 0.08032443164687006, 'SPICE': 0.2131699377828462}

    motiongpt test motiongpt:
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃         Test metric          ┃         DataLoader 0         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       Metrics/Bert_F1        │      0.3191520571708679      │
    │        Metrics/Bleu_1        │      0.4304918646812439      │
    │        Metrics/Bleu_2        │             0.0              │
    │        Metrics/Bleu_3        │             0.0              │
    │        Metrics/Bleu_4        │      0.0634666159749031      │
    │        Metrics/CIDEr         │      0.0803244337439537      │
    │    Metrics/Matching_score    │      2.9884440898895264      │
    │       Metrics/ROUGE_L        │      0.3480609357357025      │
    │  Metrics/R_precision_top_1   │      0.5357758402824402      │
    │  Metrics/R_precision_top_2   │      0.717025876045227       │
    │  Metrics/R_precision_top_3   │      0.8075430989265442      │
    │  Metrics/gt_Matching_score   │      2.9859986305236816      │
    │ Metrics/gt_R_precision_top_1 │      0.5062500238418579      │
    │ Metrics/gt_R_precision_top_2 │      0.7068965435028076      │
    │ Metrics/gt_R_precision_top_3 │      0.8004310131072998      │
    └──────────────────────────────┴──────────────────────────────┘

    motiongpt test tm2t:
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃         Test metric          ┃         DataLoader 0         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       Metrics/Bert_F1        │     0.37267205119132996      │
    │        Metrics/Bleu_1        │      0.6176044344902039      │
    │        Metrics/Bleu_2        │             0.0              │
    │        Metrics/Bleu_3        │             0.0              │
    │        Metrics/Bleu_4        │     0.21983718872070312      │
    │        Metrics/CIDEr         │      0.7112351059913635      │
    │    Metrics/Matching_score    │      2.994671583175659       │
    │       Metrics/ROUGE_L        │      0.474008172750473       │
    │  Metrics/R_precision_top_1   │      0.5245689749717712      │
    │  Metrics/R_precision_top_2   │      0.7155172228813171      │
    │  Metrics/R_precision_top_3   │      0.8144396543502808      │
    │  Metrics/gt_Matching_score   │      3.0347518920898438      │
    │ Metrics/gt_R_precision_top_1 │      0.5010775923728943      │
    │ Metrics/gt_R_precision_top_2 │      0.7021551728248596      │
    │ Metrics/gt_R_precision_top_3 │      0.790517270565033       │
    └──────────────────────────────┴──────────────────────────────┘



# Attempt 22.2: Use repeated motion code

    e30
    BLEU Score: 0.26832775461563774
    BLEU-4 Score: 0.17058964668601476
    METEOR Score: 0.3425890352057782
    ROUGE Score: 0.38217315156475296
    BERT Score: 0.8962110877037048

    {'Matching_score': tensor(4.9403), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3317), 'R_precision_top_2': tensor(0.5112), 'R_precision_top_3': tensor(0.6284), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': tensor(0.4926), 'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3733), 'CIDEr': tensor(0.0455), 
    'Bleu_4': tensor(0.0797), 
    'Bert_F1': tensor(0.3769)}

    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3216 (2): 0.1500 (3): 0.0738 (4): 0.0334
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2579
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0454
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.3768
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 4.1846 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9931 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.3808 CInt: 0.0000;(top 2) Mean: 0.5582 CInt: 0.0000;(top 3) Mean: 0.6621 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5147 CInt: 0.0000;(top 2) Mean: 0.7011 CInt: 0.0000;(top 3) Mean: 0.7978 CInt: 0.0000;

    e42
    BLEU Score: 0.25895490860130504
    BLEU-4 Score: 0.14910695957121758
    METEOR Score: 0.3223428716903407
    ROUGE Score: 0.37870007639811426
    BERT Score: 0.8956394195556641

    {'Matching_score': tensor(4.7218), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3394), 'R_precision_top_2': tensor(0.5250), 'R_precision_top_3': tensor(0.6442), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': tensor(0.4710), 'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3698), 'CIDEr': tensor(0.0579), 
    'Bleu_4': tensor(0.0797), 
    'Bert_F1': tensor(0.3816)}


    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3354 (2): 0.1607 (3): 0.0822 (4): 0.0415
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2627
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0579
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.3815
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 3.8989 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9564 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.4019 CInt: 0.0000;(top 2) Mean: 0.5800 CInt: 0.0000;(top 3) Mean: 0.6907 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5200 CInt: 0.0000;(top 2) Mean: 0.7095 CInt: 0.0000;(top 3) Mean: 0.8078 CInt: 0.0000;


    e54
    BLEU Score: 0.27431765985324336
    BLEU-4 Score: 0.1589862886299211
    METEOR Score: 0.30420150506302013
    ROUGE Score: 0.37324639453857406
    BERT Score: 0.8926103115081787

    {'Matching_score': tensor(4.2625), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3776), 'R_precision_top_2': tensor(0.5737), 'R_precision_top_3': tensor(0.6864), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': tensor(0.5141), 'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3940), 'CIDEr': tensor(0.0626), 
    'Bleu_4': tensor(0.0949), 'Bert_F1': tensor(0.4071)}

    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3517 (2): 0.1755 (3): 0.0934 (4): 0.0478
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2806
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0626
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.4069
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 3.4105 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9967 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.4733 CInt: 0.0000;(top 2) Mean: 0.6737 CInt: 0.0000;(top 3) Mean: 0.7619 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5032 CInt: 0.0000;(top 2) Mean: 0.6925 CInt: 0.0000;(top 3) Mean: 0.7950 CInt: 0.0000;

    e66
    BLEU Score: 0.26643690177710017
    BLEU-4 Score: 0.14159592969551832
    METEOR Score: 0.27383470296002177
    ROUGE Score: 0.3610452441875997
    BERT Score: 0.890285849571228

    {'Matching_score': tensor(4.1420), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3976), 'R_precision_top_2': tensor(0.5864), 'R_precision_top_3': tensor(0.6998), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': tensor(0.5213), 'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3953), 'CIDEr': tensor(0.0615), 
    'Bleu_4': tensor(0.0918), 
    'Bert_F1': tensor(0.4060)}

    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3517 (2): 0.1699 (3): 0.0865 (4): 0.0437
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2802
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0615
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.4059
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 3.2859 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9690 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.4679 CInt: 0.0000;(top 2) Mean: 0.6621 CInt: 0.0000;(top 3) Mean: 0.7603 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5134 CInt: 0.0000;(top 2) Mean: 0.7022 CInt: 0.0000;(top 3) Mean: 0.8026 CInt: 0.0000;


# Attempt 22.3: Use humanml3d only
Branch from attempt 22, and still uses padding

    e30
    BLEU Score: 0.2631338936253224
    BLEU-4 Score: 0.1571397254604008
    METEOR Score: 0.31064250090358075
    ROUGE Score: 0.3703899755573297
    BERT Score: 0.8907598853111267

    {'Matching_score': tensor(4.6625), 'gt_Matching_score': tensor(3.6208), 
    'R_precision_top_1': tensor(0.3472), 'R_precision_top_2': tensor(0.5284), 'R_precision_top_3': tensor(0.6433), 
    'gt_R_precision_top_1': tensor(0.4172), 'gt_R_precision_top_2': tensor(0.6037), 'gt_R_precision_top_3': tensor(0.7151), 
    'Bleu_1': tensor(0.4764), 'Bleu_2': tensor(0.), 'Bleu_3': tensor(0.), 
    'ROUGE_L': tensor(0.3788), 'CIDEr': tensor(0.0484), 
    'Bleu_4': tensor(0.0768), 
    'Bert_F1': tensor(0.3823)}

    ---> [M2T_EL4_DL4_NH8_PS] BLEU: (1): 0.3140 (2): 0.1403 (3): 0.0689 (4): 0.0317
    ---> [M2T_EL4_DL4_NH8_PS] ROUGE_L: 0.2571
    ---> [M2T_EL4_DL4_NH8_PS] CIDER: 0.0484
    ---> [M2T_EL4_DL4_NH8_PS] BERT SCORE: 0.3821
    ========== Matching Score Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS] Mean: 3.9012 CInterval: 0.0000
    ---> [ground truth] Mean: 2.9895 CInterval: 0.0000
    ========== R_precision Summary ==========
    ---> [M2T_EL4_DL4_NH8_PS](top 1) Mean: 0.3912 CInt: 0.0000;(top 2) Mean: 0.5741 CInt: 0.0000;(top 3) Mean: 0.6828 CInt: 0.0000;
    ---> [ground truth](top 1) Mean: 0.5099 CInt: 0.0000;(top 2) Mean: 0.6955 CInt: 0.0000;(top 3) Mean: 0.7966 CInt: 0.0000;


    e42

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃         Test metric          ┃         DataLoader 0         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       Metrics/Bert_F1        │     0.39984577894210815      │
    │        Metrics/Bleu_1        │      0.513706386089325       │
    │        Metrics/Bleu_2        │             0.0              │
    │        Metrics/Bleu_3        │             0.0              │
    │        Metrics/Bleu_4        │     0.09245716780424118      │
    │        Metrics/CIDEr         │     0.05875365063548088      │
    │    Metrics/Matching_score    │      3.4686620235443115      │
    │       Metrics/ROUGE_L        │     0.39191925525665283      │
    │  Metrics/R_precision_top_1   │     0.46918103098869324      │
    │  Metrics/R_precision_top_2   │      0.6540948152542114      │
    │  Metrics/R_precision_top_3   │      0.7607758641242981      │
    │  Metrics/gt_Matching_score   │      2.9859986305236816      │
    │ Metrics/gt_R_precision_top_1 │      0.5062500238418579      │
    │ Metrics/gt_R_precision_top_2 │      0.7068965435028076      │
    │ Metrics/gt_R_precision_top_3 │      0.8004310131072998      │
    └──────────────────────────────┴──────────────────────────────┘


    e54
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃         Test metric          ┃         DataLoader 0         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       Metrics/Bert_F1        │      0.4029705226421356      │
    │        Metrics/Bleu_1        │      0.5060538649559021      │
    │        Metrics/Bleu_2        │             0.0              │
    │        Metrics/Bleu_3        │             0.0              │
    │        Metrics/Bleu_4        │     0.08624383807182312      │
    │        Metrics/CIDEr         │     0.05646877735853195      │
    │    Metrics/Matching_score    │      3.1541950702667236      │
    │       Metrics/ROUGE_L        │     0.39340513944625854      │
    │  Metrics/R_precision_top_1   │      0.5071120858192444      │
    │  Metrics/R_precision_top_2   │      0.6978448033332825      │
    │  Metrics/R_precision_top_3   │      0.7935344576835632      │
    │  Metrics/gt_Matching_score   │      2.9859986305236816      │
    │ Metrics/gt_R_precision_top_1 │      0.5062500238418579      │
    │ Metrics/gt_R_precision_top_2 │      0.7068965435028076      │
    │ Metrics/gt_R_precision_top_3 │      0.8004310131072998      │
    └──────────────────────────────┴──────────────────────────────┘

    e66
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃         Test metric          ┃         DataLoader 0         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       Metrics/Bert_F1        │     0.40590012073516846      │
    │        Metrics/Bleu_1        │      0.489648699760437       │
    │        Metrics/Bleu_2        │             0.0              │
    │        Metrics/Bleu_3        │             0.0              │
    │        Metrics/Bleu_4        │     0.08454769849777222      │
    │        Metrics/CIDEr         │     0.06067899614572525      │
    │    Metrics/Matching_score    │      3.0756826400756836      │
    │       Metrics/ROUGE_L        │      0.3950554430484772      │
    │  Metrics/R_precision_top_1   │      0.5133620500564575      │
    │  Metrics/R_precision_top_2   │      0.690517246723175       │
    │  Metrics/R_precision_top_3   │      0.7862069010734558      │
    │  Metrics/gt_Matching_score   │      2.9859986305236816      │
    │ Metrics/gt_R_precision_top_1 │      0.5062500238418579      │
    │ Metrics/gt_R_precision_top_2 │      0.7068965435028076      │
    │ Metrics/gt_R_precision_top_3 │      0.8004310131072998      │
    └──────────────────────────────┴──────────────────────────────┘