import keras

class memory_layer(keras.layers.Layer):
    def __init__(self, memory_obj, memory_size, units, initializer="random_normal"):
        super().__init__()
        self.memory = memory_obj
        self.memory_size = memory_size
        self.units = units
        self.initailizer = initializer
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units+self.memory_size),
            initializer=self.initailizer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer=self.initailizer, trainable=True
        )
    
    def call(self, inputs):
        for i in range(inputs.shape[-1]):
            