from utils.Hasher.hasher import Hasher
a = Hasher()

print(a.padding(["Hi", "Nick"], 5, 0))

while True:
    print(a.hash_text(input("Enter text to hash: ")))