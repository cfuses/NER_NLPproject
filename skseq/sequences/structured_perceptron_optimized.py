from __future__ import division
import numpy as np
import skseq.sequences.discriminative_sequence_classifier as dsc
from skseq.sequences.sp_helpers import perceptron_update_fast
import numpy as np
from .sp_helpers import perceptron_update_fast, pad_feature_list, get_emission_features, get_transition_features, get_initial_features, get_final_features


class StructuredPerceptronOptimized(dsc.DiscriminativeSequenceClassifier):
    """
    Implements an Structured  Perceptron
    """

    def __init__(self,
                 observation_labels,
                 state_labels,
                 feature_mapper,
                 learning_rate=1.0,
                 averaged=True):

        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []
        try:
            self.parameters = np.zeros(self.feature_mapper.get_num_features())
        except Exception as e:
            print("Error allocating self.parameters:", e)
            raise

        self.fitted = False
        self.feature_mapper = feature_mapper



    def fit(self, dataset, num_epochs):
        """
        Parameters
        ----------

        dataset:
        Dataset with the sequences and tags

        num_epochs: int
        Number of epochs that the model will be trained


        Returns
        --------

        Nothing. The method only changes self.parameters.
        """
        if self.fitted:
            print("\n\tWarning: Model already trained")

        for epoch in range(num_epochs):
            acc = self.fit_epoch(dataset)
            print("Epoch: %i Accuracy: %f" % (epoch, acc))

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

        self.fitted = True

    def fit_epoch(self, dataset):
        """
        Method used to train the perceptron for a full epoch over the data

        Parameters
        ----------

        dataset:
        Dataset with the sequences and tags.

        num_epochs: int
        Number of epochs that the model will be trained


        Returns
        --------
        Accuracy for the current epoch.
        """
        num_examples = dataset.size()
        num_labels_total = 0
        num_mistakes_total = 0

        for i in range(num_examples):
            sequence = dataset.seq_list[i]
            num_labels, num_mistakes = self.perceptron_update(sequence)
            num_labels_total += num_labels
            num_mistakes_total += num_mistakes

        self.params_per_epoch.append(self.parameters.copy())
        acc = 1.0 - num_mistakes_total / num_labels_total
        return acc

    def predict_tags_given_words(self, words):
        sequence =  seq.Sequence(x=words, y=words)
        predicted_sequence, _ = self.viterbi_decode(sequence)
        return predicted_sequence.y

    def perceptron_update(self, sequence):
        predicted_sequence, _ = self.viterbi_decode(sequence)
        y_hat = np.array(predicted_sequence.y, dtype=np.int32)

        emission_features_true = []
        emission_features_hat = []
        transition_features_true = []
        transition_features_hat = []

        node_cache = self.feature_mapper.node_feature_cache
        add_emission = self.feature_mapper.add_emission_features
        edge_cache = self.feature_mapper.edge_feature_cache
        add_transition = self.feature_mapper.add_transition_features

        for i in range(len(sequence.y)):
            if sequence.y[i] != y_hat[i]:
                emission_features_true.append(
                    get_emission_features(sequence, i, sequence.y[i], node_cache, add_emission)
                )
                emission_features_hat.append(
                    get_emission_features(sequence, i, y_hat[i], node_cache, add_emission)
                )
            else:
                emission_features_true.append([])
                emission_features_hat.append([])

            if i > 0 and (sequence.y[i] != y_hat[i] or sequence.y[i - 1] != y_hat[i - 1]):
                transition_features_true.append(
                    get_transition_features(sequence, i - 1, sequence.y[i], sequence.y[i - 1], edge_cache, add_transition)
                )
                transition_features_hat.append(
                    get_transition_features(sequence, i - 1, y_hat[i], y_hat[i - 1], edge_cache, add_transition)
                )
            else:
                transition_features_true.append([])
                transition_features_hat.append([])

        # Initial and final features
        initial_cache = self.feature_mapper.initial_state_feature_cache
        add_initial = self.feature_mapper.add_initial_features
        final_cache = self.feature_mapper.final_state_feature_cache
        add_final = self.feature_mapper.add_final_features

        if sequence.y[0] != y_hat[0]:
            initial_features_true = get_initial_features(sequence, sequence.y[0], initial_cache, add_initial)
            initial_features_hat  = get_initial_features(sequence, y_hat[0], initial_cache, add_initial)
        else:
            initial_features_true = []
            initial_features_hat = []

        if sequence.y[-1] != y_hat[-1]:
            final_features_true = get_final_features(sequence, sequence.y[-1], final_cache, add_final)
            final_features_hat  = get_final_features(sequence, y_hat[-1], final_cache, add_final)
        else:
            final_features_true = []
            final_features_hat = []

        # Padding
        max_emission_len = max(
            max((len(f) for f in emission_features_true), default=0),
            max((len(f) for f in emission_features_hat), default=0)
        )
        max_transition_len = max(
            max((len(f) for f in transition_features_true), default=0),
            max((len(f) for f in transition_features_hat), default=0)
        )

        emission_features_true_padded = pad_feature_list(emission_features_true, max_emission_len)
        emission_features_hat_padded  = pad_feature_list(emission_features_hat,  max_emission_len)
        transition_features_true_padded = pad_feature_list(transition_features_true, max_transition_len)
        transition_features_hat_padded  = pad_feature_list(transition_features_hat,  max_transition_len)

        initial_features_true = np.array(initial_features_true, dtype=np.int32)
        initial_features_hat = np.array(initial_features_hat, dtype=np.int32)
        final_features_true = np.array(final_features_true, dtype=np.int32)
        final_features_hat = np.array(final_features_hat, dtype=np.int32)

        sequence_y = np.array(sequence.y, dtype=np.int32)

        # Optimized Cython update
        num_labels, num_mistakes = perceptron_update_fast(
            self.parameters,
            emission_features_true_padded,
            emission_features_hat_padded,
            transition_features_true_padded,
            transition_features_hat_padded,
            initial_features_true,
            initial_features_hat,
            final_features_true,
            final_features_hat,
            sequence_y,
            y_hat,
            self.learning_rate
        )
        return num_labels, num_mistakes

    """
    def perceptron_update(self, sequence):
        predicted_sequence, _ = self.viterbi_decode(sequence)
        y_hat = np.array(predicted_sequence.y, dtype=np.int32)

        emission_features_true = []
        emission_features_hat = []
        transition_features_true = []
        transition_features_hat = []

        for i in range(len(sequence.y)):
            if sequence.y[i] != y_hat[i]:
                emission_features_true.append(
                    self.feature_mapper.get_emission_features(sequence, i, sequence.y[i])
                )
                emission_features_hat.append(
                    self.feature_mapper.get_emission_features(sequence, i, y_hat[i])
                )
            else:
                emission_features_true.append([])
                emission_features_hat.append([])

            if i > 0 and (sequence.y[i] != y_hat[i] or sequence.y[i - 1] != y_hat[i - 1]):
                transition_features_true.append(
                    self.feature_mapper.get_transition_features(sequence, i - 1, sequence.y[i], sequence.y[i - 1])
                )
                transition_features_hat.append(
                    self.feature_mapper.get_transition_features(sequence, i - 1, y_hat[i], y_hat[i - 1])
                )
            else:
                transition_features_true.append([])
                transition_features_hat.append([])

        # Initial and final features
        if sequence.y[0] != y_hat[0]:
            initial_features_true = self.feature_mapper.get_initial_features(sequence, sequence.y[0])
            initial_features_hat = self.feature_mapper.get_initial_features(sequence, y_hat[0])
        else:
            initial_features_true = []
            initial_features_hat = []

        if sequence.y[-1] != y_hat[-1]:
            final_features_true = self.feature_mapper.get_final_features(sequence, sequence.y[-1])
            final_features_hat = self.feature_mapper.get_final_features(sequence, y_hat[-1])
        else:
            final_features_true = []
            final_features_hat = []

        max_emission_len = max(
            max(len(f) for f in emission_features_true),
            max(len(f) for f in emission_features_hat),
        )

        max_transition_len = max(
            max(len(f) for f in transition_features_true),
            max(len(f) for f in transition_features_hat),
        )


        # Convert to padded numpy arrays
        emission_features_true_padded = pad_feature_list(emission_features_true, max_emission_len)
        emission_features_hat_padded  = pad_feature_list(emission_features_hat,  max_emission_len)

        transition_features_true_padded = pad_feature_list(transition_features_true, max_transition_len)
        transition_features_hat_padded  = pad_feature_list(transition_features_hat,  max_transition_len)
        
        initial_features_true = np.array(initial_features_true, dtype=np.int32)
        initial_features_hat = np.array(initial_features_hat, dtype=np.int32)
        final_features_true = np.array(final_features_true, dtype=np.int32)
        final_features_hat = np.array(final_features_hat, dtype=np.int32)

        sequence_y = np.array(sequence.y, dtype=np.int32)
        # Optimized Cython update
        num_labels, num_mistakes = perceptron_update_fast(
            self.parameters,
            emission_features_true_padded,
            emission_features_hat_padded,
            transition_features_true_padded,
            transition_features_hat_padded,
            initial_features_true,
            initial_features_hat,
            final_features_true,
            final_features_hat,
            sequence_y,
            y_hat,
            self.learning_rate
        )
        return num_labels, num_mistakes
    
    """

    def save_model(self, dir):
        """
        Saves the parameters of the model
        """
        fn = open(dir + "parameters.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load_model(self, dir):
        """
        Loads the parameters of the model
        """
        fn = open(dir + "parameters.txt", 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
