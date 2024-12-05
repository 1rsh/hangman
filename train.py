import os
import yaml
import random
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn

from hangformer import Hangformer
from encode import encode_remaining_guesses, encode_word_state
from env import HangmanEnv

def simulate_games(model, env, num_games, device, max_length):
    samples = []
    wins = 0
    for _ in range(num_games):
        word_state, guessed, remaining_guesses = env.reset()
        done = False
        while not done:
            encoded_state = encode_word_state(word_state, max_length).to(device)
            guessed_tensor = torch.tensor(guessed).float().unsqueeze(0).to(device)
            remaining_guesses_encoded = encode_remaining_guesses(remaining_guesses)
            remaining_guesses_tensor = torch.tensor(remaining_guesses_encoded).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_probs = model(encoded_state, guessed_tensor, remaining_guesses_tensor).squeeze()
            action = torch.argmax(action_probs).item()
            letter = chr(action + ord('a'))
            next_state, done = env.step(letter)
            samples.append((word_state, guessed, remaining_guesses, env.word))
            word_state, guessed, remaining_guesses = next_state
            if done:
                if '_' not in word_state:
                    wins += 1
    accuracy = wins / num_games * 100
    return samples, accuracy

def train(model, env, optimizer, criterion, num_episodes, games_per_episode, device, max_length):
    model.to(device)
    for episode in (pbar := tqdm(range(num_episodes))):
        model.eval()
        samples, win_rate = simulate_games(model, env, games_per_episode, device, max_length)
        model.train()
        word_states, guessed_states,remaining_guesses, full_words = zip(*samples)
        encoded_states = torch.cat([encode_word_state(state, max_length) for state in word_states]).to(device)
        guessed_tensor = torch.tensor(guessed_states).float().to(device)
        remaining_guesses_encoded = [encode_remaining_guesses(guesses) for guesses in remaining_guesses]
        remaining_guesses_tensor = torch.tensor(remaining_guesses_encoded).float().to(device)
        
        targets = torch.zeros(len(full_words), 26).to(device)
        for i, (word, guessed) in enumerate(zip(full_words, guessed_states)):
            word_letters = set(word) - set('_')
            guessed_letters = set(chr(ord('a') + j) for j in range(26) if guessed[j] == 1)
            unguessed_letters = word_letters - guessed_letters
            if unguessed_letters:
                prob = 1.0 / len(unguessed_letters)
                for char in unguessed_letters:
                    targets[i, ord(char) - ord('a')] = 1
        outputs = model(encoded_states, guessed_tensor,remaining_guesses_tensor)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}, Win Rate: {win_rate:.4f}")

def test_games(model, env, device, max_length, num_samples):
    model.eval()  
    correct_guesses = 0

    sampled_words = random.sample(env.word_list, num_samples)

    for i, word in enumerate(pbar:=tqdm(sampled_words)):
        env.word = word  
        word_state, guessed,remaining_guesses = env.reset(sample=False)
        done = False

        while not done:
            encoded_state = encode_word_state(word_state, max_length).to(device)
            guessed_tensor = torch.tensor(guessed).float().unsqueeze(0).to(device)
            remaining_guesses_encoded = encode_remaining_guesses(remaining_guesses)
            remaining_guesses_tensor = torch.tensor(remaining_guesses_encoded).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_probs = model(encoded_state, guessed_tensor,remaining_guesses_tensor).squeeze()
            
            for action in torch.argsort(action_probs, descending=True):
                letter = chr(action.item() + ord('a'))
                if letter not in env.guessed_letters:
                    break

            t, done = env.step(letter)
            word_state,guessed,remaining_guesses = t
            
        if '_' not in word_state:
            correct_guesses += 1
        pbar.set_description(f"Sampling games: {correct_guesses/(i+1):.2%}")

    accuracy = correct_guesses / num_samples
    print(f"\nSample Accuracy: {accuracy:.2%}")
    return accuracy

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def main(train_txt, model_checkpoint, architecture_path, finetune, random_seed, num_epochs, num_episodes, games_per_episode, test_samples, save_dir):

    with open(train_txt, 'r') as file:
        words = file.read().split()
        
    set_seed(random_seed)

    with open(architecture_path, "r") as file:
        config = yaml.safe_load(file)

    vocab_size = config['model']['vocab_size']
    embedding_dim = config['model']['embedding_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    max_length = config['model']['max_length']

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else device)

    model = Hangformer(vocab_size, embedding_dim, num_heads, num_layers, max_length)
    
    if model_checkpoint:
        model = model.load(model_checkpoint, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    if finetune:
        for p in model.parameters():
            p.requires_grad = False

        for p in model.final_fc.parameters():
            p.requires_grad = True
    
    all_words = set(words)
    train_words = set(random.sample(words, len(words)-1000))
    test_words = all_words - train_words

    env = HangmanEnv(list(train_words))
    test_env = HangmanEnv(list(test_words))

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}\n")

        train(model, env, optimizer, criterion, num_episodes=num_episodes, games_per_episode=games_per_episode, device=device, max_length=max_length)
        test_games(model, test_env, device, max_length, test_samples)
    
        save_path = f'{save_dir}/episode_{games_per_episode*num_episodes*(i+1)}.pth'
        torch.save(model, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hangman model")
    parser.add_argument("--train_txt", type=str, default="'data/words_250000_train.txt'", help="Path to a text file containing words to train on")
    parser.add_argument("--model_checkpoint", type=str | None, default=None, help="Path to a model checkpoint")
    parser.add_argument("--architecture_path", type=str, default='architecture.yaml', help="Path to a model architecture file")
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune the model")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes per epoch")
    parser.add_argument("--games_per_episode", type=int, default=300, help="Number of games per episode")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of samples to test")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")

    args = parser.parse_args()

    main(**vars(args))