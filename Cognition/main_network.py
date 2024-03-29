from Cognition.memory_calling_layer import memory_layer
import keras


class main_network:
    def __init__(self, model=None, logger=None, exceptions=None, path="Cognition/Data/Models/"):
        self.logger=logger
        self.shoud_log = bool(logger)
        self.exceptions = exceptions
        self.should_exceptions = bool(exceptions)
        self.path = path
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model()
            
    @staticmethod
    def _generalization_layer(input_layer, size:int=50, **kwargs):
        for i in range(size//5):
            input_layer = keras.layers.Dense(max(size-i*5, size*0.2), **kwargs)(input_layer)
        for i in range(size//5):
            input_layer = keras.layers.Dense(max(i*5, size*0.2), **kwargs)(input_layer)
        return input_layer
    
    @staticmethod
    def _expansion_layer(input_layer, size:int=100, **kwargs):
        for i in range(size//5):
            input_layer = keras.layers.Dense(size+i*5, **kwargs)(input_layer)
        for i in range(size//5):
            input_layer = keras.layers.Dense(size*2-i*5, **kwargs)(input_layer)
        return input_layer
            
    def _create_model(self) -> keras.Model:
        """ Heres the way this thing should work:
        First we take in the input and original strong associtions with it
        We then split the input into two and pass it into two submodels.
            1. This one goes into a network that shrinks on the input data
            2. This one goes into one that expands on the input data
        Then we split those into 2 more each where 2 of them (one from each sub model) go through the reverse path
        Those remaining 4 inputs are then combined with a adding layer Those specific inputs are sent through memory to be queried in the concepts category
        The results of the memory query and the other 4 inputs are then sent over to a series of dense networks
        The dense newtwork is several layers of alternating sizes to run through the data and its memory
        The outputs are then split into two sections, one that gets sent to one last smaller neural network then outputed
        The other one goes into the network again for the next iteration
        20 outputs should be designated for the output of the model
        100 outputs are sent to the next iteration of the 
        Or maybe pass it all into a speaking function?
        """
        sentence_input = keras.layers.Input(shape=(500,), name="Sentece Input")
        sentecne_memory = keras.layers.Input(shape=(500,), name="Sentence memory")
        previous_output = keras.layers.Input(shape=(100), name='Previous output')
        
        all_layers = keras.layers.concatenate(axis=1)([sentence_input, sentecne_memory, previous_output])
        
        # Generalization side
        generalized = self._generalization_layer(all_layers)
        
        # Generalized and Expanded
        generalized_expanded = self._expansion_layer(generalized)
        
        # Expansion Side
        expanded = self._expansion_layer(all_layers)
        
        # Expanded and generalized
        
        expanded_generalized = self._generalization_layer(expanded)
        
        x = [generalized, generalized_expanded, expanded, expanded_generalized]
        # Large Model
        for i in range(9):
            x = keras.layers.Dense(1000-i*100)(x)
        x = keras.layers.Dense(175)(x)
        x = keras.layers.Dense(150)(x)
        x = keras.layers.Dense(140)(x)
        x = keras.layers.Dense(130)(x)
        x = keras.layers.Dense(120)(x)
        x = keras.layers.Dense(110)(x)
        x = keras.layers.Dense(100)(x)
        self.model = keras.Model(inputs=[sentence_input, sentecne_memory, previous_output], outputs=x)
        return self.model
        