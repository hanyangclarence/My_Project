bleu_1_scores = []
bleu_4_scores = []
weights_for_bleu1 = (1, 0, 0, 0)
weights_for_bleu4 = (0.25, 0.25, 0.25, 0.25)
for pred, refs in zip(prediction, reference):
    # Tokenize the prediction and references
    tokenized_pred = pred.split()
    tokenized_refs = [ref.split() for ref in refs]  # Assuming each reference is a list of sentences
    # Calculate BLEU score
    score = sentence_bleu(tokenized_refs, tokenized_pred, weights=weights_for_bleu1)
    bleu_1_scores.append(score)
    score = sentence_bleu(tokenized_refs, tokenized_pred, weights=weights_for_bleu4)
    bleu_4_scores.append(score)


rouge = Rouge()
rouge_scores = []
for pred, refs in zip(prediction, reference):
    # The `rouge` library expects a single reference, so you might need to choose how to handle multiple references
    # Here, we concatenate them, but consider the best approach for your use case
    concatenated_refs = ' '.join(refs)  # Assuming each reference is a list of sentences
    score = rouge.get_scores(pred, concatenated_refs)[0]  # get_scores returns a list of scores
    rouge_scores.append(score)
