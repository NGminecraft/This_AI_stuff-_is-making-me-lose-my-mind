class NotEnoughArgs(Exception):
    def __init__(self, expected, recieved, name):
        super().__init__(f"{name} recived to few arguments. Expected {expected}, Recieved {recieved} arguments")