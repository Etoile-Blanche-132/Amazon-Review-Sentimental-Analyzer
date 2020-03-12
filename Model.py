import keras as K
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy

def Model(wordSize):
    model = Sequential()

    model.add(Embedding(wordSize, 128))
    model.add(LSTM(128))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = RMSprop(),
                  loss = binary_crossentropy(),
                  metrics = ['accuracy'])

    return model