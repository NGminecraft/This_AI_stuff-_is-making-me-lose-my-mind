from utils.Hasher.hasher import Hasher
a = Hasher()

print(a.text_padding(["Hi", "Nick"], 5, 0))

while True:
    print(a.hash_text(input("Enter text to hash: ")))