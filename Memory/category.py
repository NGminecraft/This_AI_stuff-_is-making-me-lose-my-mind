from logging import ERROR, INFO


class Category:
    def __init__(self, id_name, objects:dict, logger=None, exceptions=None):
        self.name = id_name
        self.objects = objects
        if exceptions is not None:
            self.exceptions = exceptions
            self.skill_issues = True
        else:
            self.skill_issues = False
        if logger is not None:
            self.logger = logger
            self.should_log = True
        else:
            self.should_log = False
        self.cache = {}

    def add_item(self, item, weight):
        self.objects[item] = weight

    def category_call(self) -> dict:
        return self.objects

    def search_category(self, query=None, weight_threshold=None) -> dict:
        if query not in self.cache:
            possible = self.objects.keys()
            if query is None and weight_threshold is None and self.skill_issues:
                if self.should_log:
                    self.logger.log(ERROR, "No search criteria received")
                self.exceptions.NotEnoughArgs(1, 0, "category_search")
            elif query is not None:
                possible = [i for i in possible if query in i]
            elif weight_threshold is not None:
                possible = [i for i in possible if self.objects[i] >= weight_threshold]
            possible = sorted(possible, key=lambda x: self.objects[x], reverse=True)
            result = {key: self.objects[key] for key in possible}
            self.cache[query] = result
            return result
        else:
            # Weight threshold needs to be applied to this
            return self.cache[query]

    def update_category(self, threshold=1):
        """Removes any numbers that are too low"""
        sorted_dict = dict(sorted(self.objects.items(), key=lambda x: [1], reverse=True))
        current = 0
        while current < threshold:
            if sorted_dict[sorted_dict.keys()[-1]] < threshold:
                current = sorted_dict[sorted_dict.keys()[-1]]
                del sorted_dict[sorted_dict.keys()[-1]]
            else:
                break
                