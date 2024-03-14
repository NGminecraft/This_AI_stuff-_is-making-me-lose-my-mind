class NotEnoughArgs(Exception):
    def __init__(self, expected, received, name):
        super().__init__(f"{name} received to few arguments. Expected {expected}, Received {received} arguments")
