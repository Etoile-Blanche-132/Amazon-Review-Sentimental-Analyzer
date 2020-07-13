import keras as K
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Convolution1D, MaxPooling1D, LSTM, Dense
from keras.layers import Bidirectional
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy

def Model(setSize, inputLength):
    model = Sequential()

    model.add(Embedding(setSize, 256, input_length = inputLength))
    model.add(Dropout(0.2))
    model.add(Conv1D(256, 3, strides = 1, padding = 'valid', activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(128))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = RMSprop(learning_rate = 1e-5),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    return model
