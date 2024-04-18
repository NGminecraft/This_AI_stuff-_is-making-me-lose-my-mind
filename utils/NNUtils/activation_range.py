from tensorflow import math


class Range:
    def __init__(self, incoming_min, incoming_max, min, max):
        self.strt_min = incoming_min
        self.strt_max = incoming_max
        self.min = min
        self.max = max
        
    def RangeActivation(self, input):
        return math.add(math.divide(math.multiply(math.add(input, -self.strt_min), (self.max - self.min)), (self.strt_max - self.strt_min)), self.min)