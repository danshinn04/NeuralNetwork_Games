import json 
with open('chess_moves.json', 'r') as json_file:
    loaded_moves = json.load(json_file)

for piece, moves in loaded_moves.items():
    print(f"{piece}: {len(moves)} moves")
    print(f"Example moves: {moves[:5]}")
    print("")