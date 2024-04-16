import keras


class Loss:
    def __init__(self, logger, loss_model, weight=2, user_input=False, loss=None):
        self.logger = logger
        self.loss_model = loss_model
        self.weight = weight
        self.user_input = user_input
        self.last_loss = None
        self.past_accuracy = None
        if loss is not None:
            self.loss = loss
        else:
            self.loss = keras.losses.mean_squared_error
        
    def loss_function(self, y_true, pred):
        if self.user_input:
            inp = self._ask_input()*self.weight
        else:
            inp = self.loss_model(pred)
        current = keras.ops.add(self.loss(y_true, pred), inp)
    
    @staticmethod
    def _ask_input():
        return eval(input("Enter value from -5(horrible) to 5(perfect): "))*-1+5