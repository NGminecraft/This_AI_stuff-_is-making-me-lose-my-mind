from Cognition.memory_calling_layer import memory_layer


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
            
    def _create_model(self):
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
        """