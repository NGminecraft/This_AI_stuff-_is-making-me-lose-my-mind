def _find_letter_groups(word, chars , result=[], reverse=False):
    if word != '':
        new_part = word[0:min(chars, len(word))]
        if not reverse:
            result.append(new_part)
        else:
            result.append(new_part[::-1])
        word = word[min(chars, len(word)):]
        return _find_letter_groups(word, chars, result, reverse)
    else:
        return result

def locate_morphemes(word:str, memory_to_compare=None, letters=None, result=[]) -> list:
    if letters is None:
        letters = len(word)
    word = word.lower()
    for i in range(1, letters+1):
        result.extend(_find_letter_groups(word, i))
    result1 = []
    for i in range(1, letters+1):
        result1.extend(_find_letter_groups(word[::-1], i, reverse=True))
    result.extend(result1)
    return sorted(list(set(result)), key=lambda x: len(x))
        
"""
word = "Nicholas"
print(locate_morphemes(word, 2))
"""
def loc_in_list(list, item):
    return list.index

list = []