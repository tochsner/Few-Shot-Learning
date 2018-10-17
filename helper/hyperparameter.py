import keras

class Hyperparameter:
    def __init__(self):
        self.lr = 1
        self.r = 0
        self.batch_size = 128
        self.epochs = 20
        
        self.loss=keras.losses.categorical_crossentropy
        self.optimizer=keras.optimizers.Adam()
        self.metrics=['accuracy']