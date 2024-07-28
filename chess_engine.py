import chess.pgn

def parse_pgn(file_path):
    games = []
    counter = 1
    with open(file_path, encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            for move in game.mainline_moves():
                print(move)
            games.append(game)
            counter += 1
    return games

# Example usage
pgn_file_path = 'C:/Users/Dan/Documents/GitHub/ChessEngineOverlay/lichess_elite_2023-08.pgn'
games = parse_pgn(pgn_file_path)
