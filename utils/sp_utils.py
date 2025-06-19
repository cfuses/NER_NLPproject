from utils import sp_utils
import pandas as pd
import skseq
from skseq.sequences.sequence import Sequence
from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.structured_perceptron import StructuredPerceptron
from skseq.sequences.id_feature import IDFeatures
from seqeval.metrics import classification_report
from collections import defaultdict
import skseq.sequences.structured_perceptron as spc
from skseq.sequences import extended_feature

def evaluate_corpus(sequences, sequences_predictions):
    """Evaluate classification accuracy at corpus level, ignoring gold labels with value 0."""
    total = 0.0
    correct = 0.0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            if sequence.y[j] == 0:
                continue  # Skip evaluation for ground truth = 0
            if sequence.y[j] == y_hat:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

def show_feats(feature_mapper, seq, feature_type, inv_feature_dict):
    inv_feature_dict = {word: pos for pos, word in feature_mapper.feature_dict.items()}
    for feat,feat_ids in enumerate(feature_mapper.get_sequence_features(seq)):
        print(feature_type[feat])
        for id_list in feat_ids:
            print ("\t",id_list)
            for k,id_val in enumerate(id_list):
                print ("\t\t", inv_feature_dict[id_val] )
        print("\n")

def evaluate_model(sp, train_seq, test_seq):
    # Make predictions for the various sequences using the trained model.
    pred_train = sp.viterbi_decode_corpus(train_seq)
    pred_test = sp.viterbi_decode_corpus(test_seq)
    # Evaluate and print accuracies
    eval_train = evaluate_corpus(train_seq.seq_list, pred_train)
    eval_test = evaluate_corpus(test_seq.seq_list, pred_test)
    print("SP EXT -  Accuracy Train: %.3f Test: %.3f"%(eval_train, eval_test))

def predict_new_sentance(sp, feature_mapper, p, train_seq, feature_type, inv_feature_dict):
    word_ids  = [train_seq.x_dict[w] for w in p.split()]
    seq = skseq.sequences.sequence.Sequence(x=word_ids, y=[0 for w in word_ids])
    vit_out = sp.viterbi_decode(seq)[0]
    print(vit_out.to_words(train_seq))
    print(feature_mapper.get_sequence_features(vit_out))
    print(show_feats(feature_mapper, vit_out, feature_type, inv_feature_dict))