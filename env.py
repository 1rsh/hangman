import random

class HangmanEnv:
    def __init__(self, word_list, max_guesses=6):
        self.word_list = word_list
        self.max_guesses = max_guesses
        self.reset()

    def reset(self,sample = True):
        if sample:
            self.word = random.choice(self.word_list)
        self.guessed_letters = set()
        self.current_state = ['_'] * len(self.word)
        self.remaining_guesses = self.max_guesses
        return self.get_state()

    def step(self, letter):
        if letter in self.guessed_letters:
            self.remaining_guesses -= 1 
        else:
            self.guessed_letters.add(letter)
            if letter in self.word:
                for i, char in enumerate(self.word):
                    if char == letter:
                        self.current_state[i] = letter
            else:
                self.remaining_guesses -= 1
        
        done = '_' not in self.current_state or self.remaining_guesses <= 0
        return self.get_state(), done

    def get_state(self):
        word_state = ''.join(self.current_state)
        guessed = [1 if chr(i + ord('a')) in self.guessed_letters else 0 for i in range(26)]
        return word_state, guessed , self.remaining_guesses
