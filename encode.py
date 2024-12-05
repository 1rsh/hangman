import torch

def encode_word_state(word, max_length, vocab_size=28):
    encoding = []
    for char in word:
        if char == '_':
            encoding.append(vocab_size - 2)  # '_' is second to last
        else:
            encoding.append(ord(char) - ord('a'))  # 'a' to 'z' are 0 to 25
    
    # Pad the encoding if necessary
    if len(encoding) < max_length:
        encoding += [vocab_size - 1] * (max_length - len(encoding))  # '[PAD]' is last
    
    return torch.tensor(encoding).unsqueeze(0)

def encode_remaining_guesses(remaining_guesses, max_guesses=6):
    encoding = [0] * max_guesses
    if remaining_guesses > 0:
        encoding[max_guesses - remaining_guesses] = 1
    return encoding


def decode_word_state(arr, remove_special_tokens=False):
    array = [chr(a + ord('a')) for a in arr.tolist()]
    special_tokens = {'|': '[PAD]', '{': '[MASK]'}

    for i in range(len(array)):
        if remove_special_tokens and array[i] == '|':
            array[i] = ''
        elif array[i] in special_tokens.keys():
            array[i] = special_tokens[array[i]]
        
    return "".join(array) if remove_special_tokens else array