def longest_common_substring(s1, s2):
    # Create a 2D table to store lengths of longest common suffixes
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_pos_s1 = 0
    
    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos_s1 = i
            else:
                dp[i][j] = 0
    
    # Extract the longest common substring
    return s1[end_pos_s1 - max_length:end_pos_s1]


with open("/home/nick/vscode/python/github/AI/This_AI_stuff-_is-making-me-lose-my-mind/Memory/MorphemeTest/wordList.txt", "r") as file:
    words = list(map(lambda x: x.strip("\n"), file.readlines()))

for i in range(len(words)//2):
    print(longest_common_substring(words[i*i], words[i+1]))