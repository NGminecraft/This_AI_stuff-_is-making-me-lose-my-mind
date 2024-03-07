from logging import INFO, WARN


class Save:
    def __init__(self, logger, *args):
        self.logger = logger
        self.classes = list(args)
        
    def new_object(self, obj):
        self.classes.append(obj)
        self.logger.log(INFO, f'loaded object {type(obj).__name__}')
        
    def save(self): 
        for i in self.classes:
            try:
                i.save()
            except AttributeError:
                self.logger.log(WARN, f'Object {type(i).__name__} has no method save()')
                continue