from Memory.category import Category


class MemoryObject:
    def __init__(self, name, category:Category|list[Category]=None, autoadd=True, weight=0.1, associations:list[object]=[], assoctiation_stregnth=0.1):
        self.name = name
        self.category = category
        self.assoctiations = {}
        for i in associations:
            if i not in self.assoctiations:
                self.assoctiations[i] = assoctiation_stregnth
            else:
                self.assoctiations[i] += assoctiation_stregnth
        if autoadd:
            if type(self.category) is list:
                for i in self.category:
                    i.add_item(name, weight)
            elif type(self.category) is dict:
                for k, v in self.category:
                    k.add(name, v)
            elif type(self.category) is Category:
                self.category.add_item(self)
            self.category.add_item(name, weight)

    def get_word(self, expression:function=None) -> str|bool:
        if expression is None:
            return self.name
        else:
            return expression(self.name)
        
    def get_associated_words(self, threshold=1):
        return {k: v for k, v in sorted(self.assoctiations.keys(), key=lambda x: x[1]) if abs(v) > threshold}