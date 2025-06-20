# sp_helpers.pyx

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from skseq.sequences.label_dictionary import *
from cython cimport boundscheck, wraparound


def pad_feature_list(list feature_list_of_lists, int max_len, int pad_value=-1):
    cdef int n = len(feature_list_of_lists)
    cdef np.ndarray[np.int32_t, ndim=2] padded = np.full((n, max_len), pad_value, dtype=np.int32)

    cdef int i, j, fl_len
    cdef list fl
    for i in range(n):
        fl = feature_list_of_lists[i]
        fl_len = len(fl)
        for j in range(fl_len):
            padded[i, j] = fl[j]
    return padded

# Define C-level function
def perceptron_update_fast(double[:] parameters,
                           int[:, :] emission_features_true,
                           int[:, :] emission_features_hat,
                           int[:, :] transition_features_true,
                           int[:, :] transition_features_hat,
                           int[:] initial_features_true,
                           int[:] initial_features_hat,
                           int[:] final_features_true,
                           int[:] final_features_hat,
                           int[:] sequence_y,
                           int[:] y_hat,
                           double learning_rate):

    cdef Py_ssize_t i, j, idx
    cdef Py_ssize_t seq_len = sequence_y.shape[0]
    cdef Py_ssize_t max_feat_len = emission_features_true.shape[1]
    cdef Py_ssize_t max_transition_len = transition_features_true.shape[1]
    cdef int num_mistakes = 0

    
    # Initial state update
    for j in range(initial_features_true.shape[0]):
        idx = initial_features_true[j]
        if idx == -1:
            break
        parameters[idx] += learning_rate

    for j in range(initial_features_hat.shape[0]):
        idx = initial_features_hat[j]
        if idx == -1:
            break
        parameters[idx] -= learning_rate

    # Loop over sequence
    for i in range(seq_len):
        if sequence_y[i] != y_hat[i]:
            num_mistakes += 1

        # Emission update
        for j in range(max_feat_len):
            idx = emission_features_true[i, j]
            if idx == -1:
                break
            parameters[idx] += learning_rate

        for j in range(max_feat_len):
            idx = emission_features_hat[i, j]
            if idx == -1:
                break
            parameters[idx] -= learning_rate

        # Transition update (skip i=0)
        if i > 0:
            if sequence_y[i] != y_hat[i] or sequence_y[i - 1] != y_hat[i - 1]:
                for j in range(max_transition_len):
                    idx = transition_features_true[i, j]
                    if idx == -1:
                        break
                    parameters[idx] += learning_rate

                for j in range(max_transition_len):
                    idx = transition_features_hat[i, j]
                    if idx == -1:
                        break
                    parameters[idx] -= learning_rate

    # Final state update
    for j in range(final_features_true.shape[0]):
        idx = final_features_true[j]
        if idx == -1:
            break
        parameters[idx] += learning_rate

    for j in range(final_features_hat.shape[0]):
        idx = final_features_hat[j]
        if idx == -1:
            break
        parameters[idx] -= learning_rate

    return seq_len, num_mistakes

def get_emission_features(object sequence, int pos, int y, dict node_feature_cache, object add_emission_features):
    cdef list all_feat, node_idx
    x = sequence.x[pos]
    if x not in node_feature_cache:
        node_feature_cache[x] = {}
    if y not in node_feature_cache[x]:
        node_idx = []
        add_emission_features(sequence, pos, y, node_idx)
        node_feature_cache[x][y] = node_idx
    return node_feature_cache[x][y][:]


def get_transition_features(object sequence, int pos, int y, int y_prev, dict edge_feature_cache, object add_transition_features):
    assert 0 <= pos < len(sequence.x)
    if y not in edge_feature_cache:
        edge_feature_cache[y] = {}
    if y_prev not in edge_feature_cache[y]:
        edge_idx = []
        add_transition_features(sequence, pos, y, y_prev, edge_idx)
        edge_feature_cache[y][y_prev] = edge_idx
    return edge_feature_cache[y][y_prev]


def get_initial_features(object sequence, int y, dict initial_cache, object add_initial_features):
    if y not in initial_cache:
        edge_idx = []
        add_initial_features(sequence, y, edge_idx)
        initial_cache[y] = edge_idx
    return initial_cache[y]


def get_final_features(object sequence, int y_prev, dict final_cache, object add_final_features):
    if y_prev not in final_cache:
        edge_idx = []
        add_final_features(sequence, y_prev, edge_idx)
        final_cache[y_prev] = edge_idx
    return final_cache[y_prev]

    