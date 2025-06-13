
def print_labeled_tag_pred_example(X, y_true_tag_idxs, y_pred_tag_idxs, corpus, example_idx=0):
    """
    Displays a word-level comparison of predicted and true NER tags for a single example.

    Each word in the selected sentence is printed alongside its true and predicted tag
    in the format: word/TAG. This is useful for inspecting model performance on a
    per-sentence basis and debugging tag mismatches. 
    
    Assumes that padding tokens are labeled with "PAD0" and stops printing at the first PAD token.

    Parameters:
    X : np.ndarray
        Array of input sequences (padded word indices) of shape (num_samples, max_seq_len).
    
    y_true_tag_idxs : np.ndarray
        Array of true tag indices of shape (num_samples, max_seq_len).
    
    y_pred_tag_idxs : np.ndarray
        Array of predicted tag indices of shape (num_samples, max_seq_len).
    
    corpus : NERCorpus
        The corpus object containing `idx2word_dict` and `idx2tag_dict` mappings.

    example_idx : int, optional
        Index of the sentence to display. Default is 0.

    """

    # Extract the word indices, true tags and predicted tags for that example
    word_indices = X[example_idx]
    true_tag_indices = y_true_tag_idxs[example_idx]
    pred_tag_indices = y_pred_tag_idxs[example_idx]

    # Convert indices to words and tags
    words = [corpus.idx2word_dict[idx] for idx in word_indices]
    true_tags = [corpus.idx2tag_dict[idx] for idx in true_tag_indices]
    pred_tags = [corpus.idx2tag_dict[idx] for idx in pred_tag_indices]

    # Optional: if you have padding tokens, remove them from display
    pad_word = "PAD0"  # or idx2word[pad_index]
    pad_tag = "PAD0"   # or whatever your padding tag string is

    filtered_words = []
    filtered_true_tags = []
    filtered_pred_tags = []

    for w, t_true, t_pred in zip(words, true_tags, pred_tags):
        if w == pad_word or t_true == pad_tag:
            break
        filtered_words.append(w)
        filtered_true_tags.append(t_true)
        filtered_pred_tags.append(t_pred)

    # Format display strings
    true_seq = " ".join([f"{w}/{t}" for w, t in zip(filtered_words, filtered_true_tags)])
    pred_seq = " ".join([f"{w}/{t}" for w, t in zip(filtered_words, filtered_pred_tags)])

    print("True tags:")
    print(true_seq)
    print("\nPredicted tags:")
    print(pred_seq)