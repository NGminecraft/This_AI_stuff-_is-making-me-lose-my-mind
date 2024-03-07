from keras.preprocessing.sequence import pad_sequences

class Padder:
    def __init__(self, length:int=500, value:float=0):
        self.length = length
        self.value = value

    def pad(self, sequence:list, length_override:int=None, **kwargs):
        if length_override is None:
            sequence.extend([self.value] * (self.length - len(sequence)))
        else:
            sequence.extend([self.value] * (length_override - len(sequence)))
        return sequence