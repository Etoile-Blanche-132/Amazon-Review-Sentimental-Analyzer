import matplotlib.pyplot as plt
from skeras import plot_acc, plot_loss
from keras.models import load_model
from keras.callbacks import EarlyStopping

from Model import Model
from Dataset import DATA

data = DATA()
model = Model(data.setSize, data.inputLength)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)

if __name__ == "__main__":
    batchSize = 128
    epochs = int(input('Epochs: '))

    history = model.fit(data.X_train, data.y_train,
                        batch_size = batchSize,
                        epochs = epochs,
                        validation_split = 0.2,
                        callbacks = [es])
              
    score = model.evaluate(data.X_train, data.y_train)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()

    model.save('Amazon_Review_Sentiment_Analyzer.h5')
