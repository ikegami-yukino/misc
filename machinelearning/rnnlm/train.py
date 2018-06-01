import numpy as np
import joblib
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM

MAX_LENGTH = 40
EPOCHS = 3
EMBEDDING_OUT_DIM = 650
LSTM_UNITS = 650
DROPOUT_RATE = 0.5

# prepare the tokenizer on the source text
with open('train.txt') as fd:
    data = fd.read()
tokenizer = Tokenizer(split='\t')
tokenizer.fit_on_texts([data])
index_to_word = {index: word for word, index in tokenizer.word_index.items()}
joblib.dump(tokenizer, 'tokenizer.pkl', compress=True)

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create line-based sequences
sequences = list()
for line in data.splitlines():
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
del data
print('Total Sequences: %d' % len(sequences))

# pad input sequences
sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='pre')

# split into input and output elements
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
del sequences
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_OUT_DIM, input_length=MAX_LENGTH-1))
model.add(LSTM(LSTM_UNITS))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(X, y, epochs=EPOCHS, verbose=2)

# Save model to file
open('rnnlm.yaml', 'w').write(model.to_yaml())
model.save_weights('rnnlm.hdf5')
