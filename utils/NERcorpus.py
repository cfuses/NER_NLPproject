#%%
import csv
from collections import defaultdict
from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary
#%%

class NERCorpus(object):
    """
    Reads a NER dataset and stores a SequenceList and dictionaries.
    """
    def __init__(self):
        self.word_dict = LabelDictionary()
        self.tag_dict = LabelDictionary()
        self.sequence_list = SequenceList(self.word_dict, self.tag_dict)

    def read_sequence_list_csv(self, filename, max_sent_len=100000, max_nr_sent=100000):
        """
        Reads NER data from a CSV with columns: sentence_id, words, tags
        and builds a SequenceList
        """
        instance_list = self.read_ner_instances(filename, max_sent_len, max_nr_sent)

        # Step 1: Populate dictionaries
        for sent_x, sent_y in instance_list:
            for word in sent_x:
                if word not in self.word_dict:
                    self.word_dict.add(word)
            for tag in sent_y:
                if tag not in self.tag_dict:
                    self.tag_dict.add(tag)

        # Step 2: Build the sequence list
        seq_list = SequenceList(self.word_dict, self.tag_dict)
        for sent_x, sent_y in instance_list:
            seq_list.add_sequence(sent_x, sent_y, self.word_dict, self.tag_dict)

        return seq_list

    def read_ner_instances(self, file_path, max_sent_len=100000, max_nr_sent=100000):
        """
        Reads CSV data in sentence_id,words,tags format and returns list of (x, y) sequences
        """
        sentences = defaultdict(list)

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sent_id = int(row["sentence_id"])
                word = row["words"]
                tag = row["tags"]
                sentences[sent_id].append((word, tag))

        instance_list = []
        for sentence in sentences.values():
            if len(sentence) == 0:
                continue
            words, tags = zip(*sentence)
            instance_list.append((list(words), list(tags)))
            if len(instance_list) >= max_nr_sent:
                break

        return instance_list
# %%
