import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import train

TRAIN_PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'train.tsv'))
VALID_PATH = os.path.join(os.path.join(os.path.dirname(__file__), 'valid.tsv'))


def test_tokenize():
    tokenizer = train.tokenize(TRAIN_PATH, VALID_PATH, delimiter='\t')
    test_data = [['容疑者', 'が']]
    assert tokenizer.texts_to_sequences(test_data) == [[265, 3]]


class TestDataGenerator():
    def setup(self):
        tokenizer = train.tokenize(TRAIN_PATH, VALID_PATH, delimiter='\t')
        batch_size = 3
        max_length = 20
        vocab_size = 864
        self.dg = train.DataGenerator(TRAIN_PATH, batch_size, max_length, vocab_size, tokenizer)

    def test___len__(self):
        assert len(self.dg) == 34

    def test___getitem__(self):
        X, y = self.dg.__getitem__(0)
        assert X.shape == (3, 19)
        assert y.shape == (3, 19, 864)
