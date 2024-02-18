import pandas as pd

def format(word):
    word = word.lower()
    item = []
    for i in word:
        if i.isnumeric() or (ord(i) >= 97 and ord(i) <=122):
            item.append(i)
    return ''.join(item)
    
    
x = pd.read_csv("data_train.csv")
result = []
for i in x["Text"]:
    i = i.split(' ')
    for j in i:
        j = format(j)
        if j not in result:
            result.append(j)
        
y = pd.read_csv("data_test.csv")
for i in y["Text"]:
    i = i.split(' ')
    for j in i:
        j = format(j)
        if j not in result:
            result.append(j)

print(result)
print()
print(len(result))