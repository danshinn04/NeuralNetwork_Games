import chess
import chess.pgn
import json

def modify_move_notation(board, move):
    piece = board.piece_at(move.from_square)
    piece_symbol = piece.symbol().upper() if piece.color else piece.symbol().lower()
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    
    # Handle pawn captures separately to include the capture notation
    if piece_symbol.lower() == 'p':
        if board.is_capture(move):
            return f"{from_square[0]}x{to_square}"
        return f"{from_square}{to_square}"
    
    # Include piece type for other moves
    return f"{from_square}{piece_symbol}{to_square}"

def process_pgn_moves(file_path, max_games=20):
    pgn_moves_list = []
    with open(file_path, encoding='utf-8') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        game_count = 0
        while game and game_count < max_games:
            board = game.board()
            modified_moves = []
            for move in game.mainline_moves():
                modified_move = modify_move_notation(board, move)
                modified_moves.append(modified_move)
                board.push(move)
                # Debug: Print board state after each move
                print(board)
            pgn_moves_list.append(modified_moves)
            print(f"Game {game_count + 1} PGN moves:", modified_moves)
            game = chess.pgn.read_game(pgn_file)
            game_count += 1
    return pgn_moves_list

def load_moves_dict(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)

def match_moves_to_dict(pgn_moves, moves_dict):
    match_results = []
    for game_moves in pgn_moves:
        game_match_results = []
        for move in game_moves:
            found_match = False
            for piece, piece_moves in moves_dict.items():
                
                
                # Check for pawn captures and regular moves
                if piece.lower() == 'p':
                    if move in piece_moves or move.replace('x', '') in piece_moves:
                        game_match_results.append((move, piece))
                        found_match = True
                        break
                else:
                    if move in piece_moves:
                        game_match_results.append((move, piece))
                        found_match = True
                        break
            if not found_match:
                game_match_results.append((move, "Not Found"))
        match_results.append(game_match_results)
    return match_results

pgn_file_path = 'C:/Users/Dan/Documents/GitHub/ChessEngineOverlay/lichess_elite_2023-08.pgn'
json_file_path = 'chess_moves.json'
pgn_moves = process_pgn_moves(pgn_file_path)
moves_dict = load_moves_dict(json_file_path)
print("Moves for black pieces in the dictionary:")
for piece, moves in moves_dict.items():
    if piece.islower():
        print(f"{piece}: {moves[:5]}")  # Print the first 5 moves for each black piece

# Match PGN moves to dictionary
match_results = match_moves_to_dict(pgn_moves, moves_dict)

# Print match results
for game_index, game_result in enumerate(match_results):
    print(f"Game {game_index + 1} match results:")
    for move, piece in game_result:
        print(f"Move: {move}, Piece: {piece}")
    print("")

# Save reformatted moves dictionary
reformatted_dict = match_moves_to_dict(pgn_moves, moves_dict)
with open('reformatted_chess_moves.json', 'w') as json_file:
    json.dump(reformatted_dict, json_file)

print("Reformatted moves dictionary saved to reformatted_chess_moves.json")
