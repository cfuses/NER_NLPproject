from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):
    
    def get_emission_features(self, sequence, pos, y):
        features = []
        # Call your add_emission_features to add features to the list
        self.add_emission_features(sequence, pos, y, features)
        return features
    
    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        y_name = self.dataset.y_dict.get_label_name(y)

        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)

        # Feature 1: id
        feat_id = self.add_feature(f"id:{word}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Feature 2: lowercased word
        feat_id = self.add_feature(f"lower:{word.lower()}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Feature 3: capitalized
        if word[0].isupper():
            feat_id = self.add_feature(f"capitalized::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Feature 4: all caps
        if word.isupper():
            feat_id = self.add_feature(f"allcaps::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Feature 5: is digit
        if word.isdigit():
            feat_id = self.add_feature(f"isdigit::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Feature 6: suffix/prefix
        if len(word) >= 3:
            prefix = word[:3].lower()
            suffix = word[-3:].lower()

            feat_id = self.add_feature(f"prefix:{prefix}::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

            feat_id = self.add_feature(f"suffix:{suffix}::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        if pos == 0:
            pos_bucket = "start"
        elif pos == len(sequence.x) - 1:
            pos_bucket = "end"
        elif pos < 3:
            pos_bucket = "early"
        elif pos > len(sequence.x) - 4:
            pos_bucket = "late"
        else:
            pos_bucket = "middle"

        feat_id = self.add_feature(f"pos_bucket:{pos_bucket}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)


        return features
