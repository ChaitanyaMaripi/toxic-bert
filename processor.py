""" 
@Author: Kumar Nityan Suman
@Date: 2019-05-19 12:47:13
"""

# Load packages
import os
import csv
import pandas as pd
import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = list()
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = list()
            for line in reader:
                lines.append(line)
            return lines


class ToxicProcessor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, filename, data_dir):
        """See base class."""
        try:
            lines = self._read_csv(os.path.join(data_dir, "train.csv"))
            examples = self._create_examples(filename, lines, "train")
        except Exception as e:
            print(e)
    
    def get_test_examples(self, filename, data_dir):
        """See base class."""
        try:
            lines = self._read_csv(os.path.join(data_dir, "test.csv"))
            examples = self._create_examples(filename, lines, "test")
        except Exception as e:
            print(e)

    def get_labels(self, filename):
        """See base class."""
            return [0, 1]

    def _create_examples(self, filename, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = list()
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            if set_type == "test":
                label = None
            else:
                label = tokenization.convert_to_unicode(str(line[1]))
            # print("Text_a: {} -> Text_b: {} -> Label: {}\n".format(text_a, text_b, label))
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples



