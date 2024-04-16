from keras.preprocessing.sequence import pad_sequences

class Padder:
    def __init__(self, length:int=500, value:float=0):
        self.length = length
        self.value = value

    def pad(self, sequence:list, length_override:int=None, **kwargs):
        if length_override is None:
            length_override = self.length
        sequence.extend([self.value] * (length_override - len(sequence)))
        assert len(sequence) == length_override, f"Padder length is {len(sequence)}. It should be {length_override}"
        return sequence
