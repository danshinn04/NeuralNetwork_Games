import chess
import json

def generate_all_moves():
    piece_types = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    moves_dict = {piece: [] for piece in piece_types}
    for square in chess.SQUARES:
        count = 0
        for piece in piece_types:
            count += 1
            board = chess.Board()
            board.clear()
            piece_obj = chess.Piece.from_symbol(piece)
            board.set_piece_at(square, piece_obj)
            
            board.turn = piece_obj.color
            
            for move in board.legal_moves:
                if board.piece_at(move.from_square).symbol() == piece:
                    if count > 6:
                        moves_dict[piece].append(f"{chess.square_name(move.from_square)}{board.san(move).lower()}")
                    else:
                        moves_dict[piece].append(f"{chess.square_name(move.from_square)}{board.san(move)}")
    
    add_castling_moves(moves_dict)
    add_pawn_capture_moves(moves_dict)

    return moves_dict

def add_castling_moves(moves_dict):
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece.from_symbol('K'))
    board.set_piece_at(chess.H1, chess.Piece.from_symbol('R'))
    board.set_piece_at(chess.A1, chess.Piece.from_symbol('R'))
    board.set_castling_fen('KQ')
    board.turn = chess.WHITE
    for move in board.legal_moves:
        if board.is_castling(move):
            moves_dict['K'].append(f"{chess.square_name(move.from_square)}{board.san(move)}")

    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E8, chess.Piece.from_symbol('k'))
    board.set_piece_at(chess.H8, chess.Piece.from_symbol('r'))
    board.set_piece_at(chess.A8, chess.Piece.from_symbol('r'))
    board.set_castling_fen('kq')
    board.turn = chess.BLACK
    for move in board.legal_moves:
        if board.is_castling(move):
            moves_dict['k'].append(f"{chess.square_name(move.from_square)}{board.san(move)}")

def add_pawn_capture_moves(moves_dict):
    for square in chess.SQUARES:
        board = chess.Board()
        board.clear()
        board.set_piece_at(square, chess.Piece.from_symbol('P'))
        board.turn = chess.WHITE
        
        for file_offset in [-1, 1]:
            capture_square = square + file_offset + 8
            if chess.SQUARES[0] <= capture_square <= chess.SQUARES[63] and chess.square_file(capture_square) in range(8):
                board.set_piece_at(capture_square, chess.Piece.from_symbol('p'))
                for move in board.legal_moves:
                    if board.piece_at(move.from_square).symbol() == 'P' and move.to_square == capture_square:
                        moves_dict['P'].append(f"{chess.square_name(move.from_square)}{board.san(move)}")
                board.remove_piece_at(capture_square)

        if chess.square_rank(square) == 6:
            for promotion_piece in ['N', 'B', 'R', 'Q']:
                for file_offset in [-1, 1]:
                    capture_square = square + file_offset + 8
                    if chess.SQUARES[0] <= capture_square <= chess.SQUARES[63] and chess.square_file(capture_square) in range(8):
                        board.set_piece_at(capture_square, chess.Piece.from_symbol('p'))
                        for move in board.legal_moves:
                            if board.piece_at(move.from_square).symbol() == 'P' and move.to_square == capture_square:
                                san_move = board.san(move).replace('=', '') + f"={promotion_piece}"
                                moves_dict['P'].append(f"{chess.square_name(move.from_square)}{san_move}")
                        board.remove_piece_at(capture_square)

    for square in chess.SQUARES:
        board = chess.Board()
        board.clear()
        board.set_piece_at(square, chess.Piece.from_symbol('p'))
        board.turn = chess.BLACK
        
        for file_offset in [-1, 1]:
            capture_square = square + file_offset - 8
            if chess.SQUARES[0] <= capture_square <= chess.SQUARES[63] and chess.square_file(capture_square) in range(8):
                board.set_piece_at(capture_square, chess.Piece.from_symbol('P'))
                for move in board.legal_moves:
                    if board.piece_at(move.from_square).symbol() == 'p' and move.to_square == capture_square:
                        moves_dict['p'].append(f"{chess.square_name(move.from_square)}{board.san(move)}")
                board.remove_piece_at(capture_square)

        if chess.square_rank(square) == 1:
            for promotion_piece in ['n', 'b', 'r', 'q']:
                for file_offset in [-1, 1]:
                    capture_square = square + file_offset - 8
                    if chess.SQUARES[0] <= capture_square <= chess.SQUARES[63] and chess.square_file(capture_square) in range(8):
                        board.set_piece_at(capture_square, chess.Piece.from_symbol('P'))
                        for move in board.legal_moves:
                            if board.piece_at(move.from_square).symbol() == 'p' and move.to_square == capture_square:
                                san_move = board.san(move).replace('=', '') + f"={promotion_piece}"
                                moves_dict['p'].append(f"{chess.square_name(move.from_square)}{san_move}")
                        board.remove_piece_at(capture_square)

all_moves = generate_all_moves()
for piece, moves in all_moves.items():
    print(f"{piece}: {len(moves)} moves")
    print(f"Example moves: {moves[:]}")
    print("")

with open('chess_moves.json', 'w') as json_file:
    json.dump(all_moves, json_file)

print("Moves dictionary saved to chess_moves.json")
