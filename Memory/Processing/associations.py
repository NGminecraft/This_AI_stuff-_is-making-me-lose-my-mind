class Associations:
    def __init__(self, logger, exceptions, memory):
        self.logger = logger
        self.exceptions = exceptions
        self. memory = memory
        
    def update_assoctiations(self, sentence_input:str) -> bool:
        if ' ' in sentence_input:
            sentence_input = sentence_input.split(' ')
        else:
            sentence_input = [sentence_input]
        for i in sentence_input:
            pass