# sp_helpers.pyx
import numpy as np
cimport numpy as np
# sp_helpers.pyx
import numpy as np
cimport numpy as np

def perceptron_update_fast(np.ndarray[np.float64_t, ndim=1] parameters,
                           list emission_features_true,
                           list emission_features_hat,
                           list transition_features_true,
                           list transition_features_hat,
                           list initial_features_true,
                           list initial_features_hat,
                           list final_features_true,
                           list final_features_hat,
                           list sequence_y,
                           list y_hat,
                           double learning_rate):
    """
    Fast version of the perceptron update, avoiding feature_mapper calls inside Cython.

    Parameters:
        - parameters: weight vector to update
        - *_features_*: precomputed feature indices (Python lists of lists)
        - sequence_y: true labels (Python list)
        - y_hat: predicted labels (Python list)
        - learning_rate: scalar float
    Returns:
        - num_labels: total number of labels (int)
        - num_mistakes: number of label mismatches (int)
    """
    cdef int i, idx
    cdef int num_labels = len(sequence_y)
    cdef int num_mistakes = 0

    # Count mistakes
    for i in range(num_labels):
        if sequence_y[i] != y_hat[i]:
            num_mistakes += 1

    # Initial state update
    for idx in initial_features_true:
        parameters[idx] += learning_rate
    for idx in initial_features_hat:
        parameters[idx] -= learning_rate

    # Emissions and transitions
    for i in range(num_labels):
        for idx in emission_features_true[i]:
            parameters[idx] += learning_rate
        for idx in emission_features_hat[i]:
            parameters[idx] -= learning_rate

        for idx in transition_features_true[i]:
            parameters[idx] += learning_rate
        for idx in transition_features_hat[i]:
            parameters[idx] -= learning_rate

    # Final state update
    for idx in final_features_true:
        parameters[idx] += learning_rate
    for idx in final_features_hat:
        parameters[idx] -= learning_rate

    return num_labels, num_mistakes



def perceptron_update_cython(np.ndarray[np.float64_t, ndim=1] parameters,
                             object sequence,             # <-- full Sequence object
                             np.ndarray sequence_y,
                             np.ndarray y_hat,
                             object feature_mapper,
                             double learning_rate):

    """
    Cythonized version of the perceptron_update core loop.
    Only do numpy array operations and direct C-level loops here.
    feature_mapper calls stay as Python calls.

    Returns: (num_labels, num_mistakes) tuple
    """
    cdef int num_labels = 0
    cdef int num_mistakes = 0
    cdef int pos, seq_len = len(sequence_y)

    # Initial features update
    if sequence_y[0] != y_hat[0]:
        true_initial_features = feature_mapper.get_initial_features(sequence, sequence_y[0])
        for idx in true_initial_features:
            parameters[idx] += learning_rate
        hat_initial_features = feature_mapper.get_initial_features(sequence, y_hat[0])
        for idx in hat_initial_features:
            parameters[idx] -= learning_rate

    for pos in range(seq_len):
        y_t_true = sequence_y[pos]
        y_t_hat = y_hat[pos]
        num_labels += 1
        if y_t_true != y_t_hat:
            num_mistakes += 1
            true_emission_features = feature_mapper.get_emission_features(sequence, pos, y_t_true)
            for idx in true_emission_features:
                parameters[idx] += learning_rate
            hat_emission_features = feature_mapper.get_emission_features(sequence, pos, y_t_hat)
            for idx in hat_emission_features:
                parameters[idx] -= learning_rate

        if pos > 0:
            prev_y_t_true = sequence_y[pos-1]
            prev_y_t_hat = y_hat[pos-1]
            if y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat:
                # Note the order of arguments here: pos-1, prev_label, curr_label
                true_transition_features = feature_mapper.get_transition_features(sequence, pos-1, prev_y_t_true, y_t_true)
                for idx in true_transition_features:
                    parameters[idx] += learning_rate
                hat_transition_features = feature_mapper.get_transition_features(sequence, pos-1, prev_y_t_hat, y_t_hat)
                for idx in hat_transition_features:
                    parameters[idx] -= learning_rate

    # Final features update
    y_t_true = sequence_y[seq_len-1]
    y_t_hat = y_hat[seq_len-1]
    if y_t_true != y_t_hat:
        true_final_features = feature_mapper.get_final_features(sequence, y_t_true)
        for idx in true_final_features:
            parameters[idx] += learning_rate
        hat_final_features = feature_mapper.get_final_features(sequence, y_t_hat)
        for idx in hat_final_features:
            parameters[idx] -= learning_rate

    return num_labels, num_mistakes
