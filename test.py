import keras

b= ["hello", "hi", "Nick", "Cow"]
token = keras.preprocessing.text.Tokenizer(oov_token=-1, split=' ')
token.fit_on_texts(b)
while True:
    c = [input("Enter ")]
    print(c)
    b = token.texts_to_sequences(c)
    print(b, "tokenized")
    print(len(b), "length")
    a = keras.preprocessing.sequence.pad_sequences(b, maxlen=5, padding='post', truncating='post')
    print(a, "padded")
    print(len(a), "length of padded")
    print(a.shape, "shape of padded")