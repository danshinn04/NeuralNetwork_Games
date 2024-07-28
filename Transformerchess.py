import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import chess.pgn
import random

def read_pgn(file_path, max_games=10000):
    games = []
    pgn = open(file_path, encoding='utf-8')
    while True:
        if len(games) >= max_games:
            break
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        moves = [move.uci() for move in game.mainline_moves()]
        games.append(moves)
    pgn.close()
    return games

def create_tokenizer(games):
    move_to_idx = {}
    idx_to_move = {}
    all_moves = set(move for game in games for move in game)
    for idx, move in enumerate(sorted(all_moves)):
        move_to_idx[move] = idx
        idx_to_move[idx] = move
    return move_to_idx, idx_to_move

class ChessDataset(Dataset):
    def __init__(self, games, move_to_idx):
        self.games = [torch.tensor([move_to_idx[move] for move in game], dtype=torch.long) for game in games]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        return game[:-1], game[1:]

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * np.sqrt(d_model)
        tgt = self.embedding(tgt) * np.sqrt(d_model)
        output = self.transformer(src, tgt)
        return self.output_linear(output)

def train(model, data_loader, optimizer, loss_fn, epochs, device='cpu'):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(data_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            output_flat = output.view(-1, output.size(-1))
            loss = loss_fn(output_flat, tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Current Loss: {loss.item()}')
        print(f'Epoch {epoch+1}: Average Loss = {total_loss / len(data_loader)}')

# f(prev) -> dec(enc(f(move)))
def model_select_move(model, board, move_to_idx, idx_to_move, device='cpu'):
    model.to(device)
    model.eval()
    tokens = [move_to_idx[move.uci()] for move in board.move_stack]
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(tokens_tensor, tokens_tensor)
        probabilities = torch.softmax(output[0, -1], dim=0)
        move_idx = torch.argmax(probabilities).item()
        predicted_move = idx_to_move[move_idx]

    if predicted_move in [move.uci() for move in board.legal_moves]:
        return predicted_move
    else:
        return random.choice([move.uci() for move in board.legal_moves])

# Play game against the model
def play_against_model(model, move_to_idx, idx_to_move, device='cpu'):
    board = chess.Board()
    print("Initial board position:\n", board, "\n")

    while not board.is_game_over():
        print("Current board position:\n", board, "\n")
        
        if board.turn == chess.WHITE:
            print("Your move (you are playing as White). Enter move in UCI format (e.g., e2e4): ")
            user_move = input()
            if user_move in [move.uci() for move in board.legal_moves]:
                board.push_uci(user_move)
            else:
                print("Illegal move, try again.")
                continue
        else:
            print("Model is thinking...")
            model_move = model_select_move(model, board, move_to_idx, idx_to_move, device)
            board.push_uci(model_move)
            print(f"Model's move: {model_move}")
    
    print("Game over. Result:", board.result())


if __name__ == "__main__":
    file_path = 'C:/Users/Dan/Downloads/lichess_elite_2023-07/lichess_elite_2023-07.pgn'

    games = read_pgn(file_path, max_games=10000)
    move_to_idx, idx_to_move = create_tokenizer(games)

    dataset = ChessDataset(games, move_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

    vocab_size = len(move_to_idx)
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, loss_fn, epochs=10)
    torch.save(model, 'chess_transformer_model_full.pth')
    
    play_against_model(model, move_to_idx, idx_to_move)
