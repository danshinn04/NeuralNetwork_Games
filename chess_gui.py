import chess
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")
        
        self.board = chess.Board()
        
        self.canvas = tk.Canvas(self.root, width=480, height=480)
        self.canvas.pack()
        
        self.status_label = ttk.Label(self.root, text="White to move")
        self.status_label.pack()
        
        self.piece_images = self.load_piece_images()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.click)
        
        self.selected_square = None

    def load_piece_images(self):
        pieces = ['wp', 'bp', 'wn', 'bn', 'wb', 'bb', 'wr', 'br', 'wq', 'bq', 'wk', 'bk']
        piece_images = {}
        for piece in pieces:
            image = Image.open(f"pieces/{piece}.png")
            image = image.resize((60, 60), Image.LANCZOS)
            piece_images[piece] = ImageTk.PhotoImage(image)
        return piece_images

    def evaluate(self):
        position = self.board
    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#A66D4F", "#DDB88C"]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                x1 = col * 60
                y1 = (7 - row) * 60
                x2 = x1 + 60
                y2 = y1 + 60
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                
                piece = self.board.piece_at(chess.square(col, row))
                if piece:
                    piece_symbol = piece.symbol()
                    piece_color = 'w' if piece_symbol.isupper() else 'b'
                    piece_type = piece_symbol.lower()
                    piece_key = f"{piece_color}{piece_type}"
                    self.canvas.create_image(x1 + 30, y1 + 30, image=self.piece_images[piece_key])

    def click(self, event):
        col = event.x // 60
        row = 7 - (event.y // 60)
        square = chess.square(col, row)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.highlight_square(square)
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_status()
            self.selected_square = None
            self.draw_board()

    def highlight_square(self, square):
        x = chess.square_file(square) * 60
        y = (7 - chess.square_rank(square)) * 60
        self.canvas.create_rectangle(x, y, x + 60, y + 60, outline="red", width=3)

    def update_status(self):
        if self.board.is_checkmate():
            self.status_label.config(text="Checkmate! {} wins".format("White" if self.board.turn else "Black"))
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate!")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw by insufficient material")
        else:
            self.status_label.config(text="{} to move".format("White" if self.board.turn else "Black"))

root = tk.Tk()
app = ChessGUI(root)
root.mainloop()
