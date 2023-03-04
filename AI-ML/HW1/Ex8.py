# Ex8: Let user type 2 words in English as input. Print out the output
# which is the shortest chain according to the following rules:
# - Each word in the chain has at least 3 letters
# - The 2 input words from user will be used as the first and the last words of the chain
# - 2 last letters of 1 word will be the same as 2 first letters of the next word in the chain
# - All the words are from the file wordsEn.txt
# - If there are multiple shortest chains, return any of them is sufficient

import pandas as pd
def load_words():
    with open('wordEn.txt', 'r') as f:
        words = f.read().splitlines()
    return set(words)

def find_shortest_chain(words, start_word, end_word):
    if len(start_word) != len(end_word):
        return None

    # Chỉ giữ lại các từ có độ dài lớn hơn hoặc bằng 3
    words = [word for word in words if len(word) >= 3]

    # Tạo một tập hợp chứa các từ đã sử dụng
    used_words = set([start_word])

    # Tạo một danh sách chứa các chuỗi ngắn nhất tìm được
    shortest_chains = [[start_word]]

    # Lặp lại cho đến khi tìm thấy chuỗi ngắn nhất hoặc không còn từ nào khả dụng
    while len(shortest_chains) > 0:
        # Lấy chuỗi ngắn nhất trong danh sách
        shortest_chain = shortest_chains.pop(0)

        # Nếu từ cuối cùng trong chuỗi là từ kết thúc, trả lại chuỗi ngắn nhất
        if shortest_chain[-1] == end_word:
            return shortest_chain

        # Lấy 2 chữ cái cuối cùng của từ cuối cùng trong chuỗi
        last_two_chars = shortest_chain[-1][-2:]

        # Lặp lại qua tất cả các từ trong tập hợp từ
        for word in words:
            # Nếu từ chưa được sử dụng và có cùng 2 chữ cái đầu tiên với 2 chữ cái cuối cùng của từ cuối cùng trong chuỗi
            if word not in used_words and word[:2] == last_two_chars:
                # Thêm từ vào tập hợp các từ đã sử dụng
                used_words.add(word)
                # Tạo một chuỗi mới bằng cách sao chép chuỗi ngắn nhất hiện tại và thêm từ mới vào
                new_chain = shortest_chain.copy()
                new_chain.append(word)
                # Thêm chuỗi mới vào danh sách các chuỗi ngắn nhất tìm được
                shortest_chains.append(new_chain)

    return None

# Hàm main để chạy chương trình
def main():
    # Nhập vào 2 từ đầu vào từ người dùng
    start_word = input("Nhập từ đầu tiên: ").lower()
    end_word = input("Nhập từ cuối cùng: ").lower()

    # Tải danh sách các từ từ tệp tin
    words = load_words()

    # Tìm chuỗi ngắn nhất
    shortest_chain = find_shortest_chain(words, start_word, end_word)

    # In kết quả
    if shortest_chain:
        print("Chuỗi ngắn nhất là:", "->".join(shortest_chain))
    else:
        print("Không tìm thấy chuỗi ngắn nhất")

# Chạy hàm main
if __name__ == '__main__':
    main()



