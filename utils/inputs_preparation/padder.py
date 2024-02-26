from keras.preprocessing.sequence import pad_sequences

class Padder:
    def __init__(self, length:int=500, value:float=0):
        self.length = length
        self.value = value

    def pad(self, sequence:list, length_override:int=None, *args:"Extra arguments to pad with"):
        if length_override is not None:
            return pad_sequences(sequence, maxlen=length_override, *args)
        elif length_override is None and len(args) == 0:
            return pad_sequences(sequence, maxlen=self.length, truncating='post', padding='post', value=self.value)
        else:
            return pad_sequences(sequence, maxlen=self.length, *args)