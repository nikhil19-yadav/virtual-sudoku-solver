import cv2
import numpy as np

#  Sudoku Puzzles (3 puzzles now)
puzzles = [
    # Puzzle 1
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    # Puzzle 2
    [
        [0, 2, 0, 6, 0, 8, 0, 0, 0],
        [5, 8, 0, 0, 0, 9, 7, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0],
        [3, 7, 0, 0, 0, 0, 5, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 8, 0, 0, 0, 0, 1, 3],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 9, 8, 0, 0, 0, 3, 6],
        [0, 0, 0, 3, 0, 6, 0, 9, 0]
    ],
    # Puzzle 3
    [
        [0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 8, 0, 0, 0, 7, 0, 9, 0],
        [6, 0, 2, 0, 0, 0, 5, 0, 0],
        [0, 7, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 9, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 4, 0],
        [0, 0, 5, 0, 0, 0, 6, 0, 3],
        [0, 9, 0, 4, 0, 0, 0, 7, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0]
    ]
]

selected_cell = None
puzzle_index = 0
h, w = 720, 1280
grid_size = min(h, w) - 100
start_x = (w - grid_size) // 2
start_y = (h - grid_size) // 2
cell_size = grid_size // 9

# Track per-cell colors { (row,col): (B,G,R) }
cell_colors = {}

# Number pad positions
button_size = 50
margin = 15
pad_start_x = w - (button_size * 3) - (margin * 4)
pad_start_y = (h - (button_size * 4) - (margin * 3)) // 2
num_pad_buttons = {}

def is_valid_move(board, row, col, num):
    """Return True if num can be placed at (row,col) without Sudoku conflict."""
    # Check row
    for j in range(9):
        if j != col and board[row][j] == num:
            return False
    # Check column
    for i in range(9):
        if i != row and board[i][col] == num:
            return False
    # Check 3x3 box
    br, bc = (row//3)*3, (col//3)*3
    for i in range(br, br+3):
        for j in range(bc, bc+3):
            if (i != row or j != col) and board[i][j] == num:
                return False
    return True

def draw_sudoku_grid(image, board):
    for i in range(10):
        thick = 3 if i % 3 == 0 else 1
        cv2.line(image, (start_x + i * cell_size, start_y),
                 (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), thick)
        cv2.line(image, (start_x, start_y + i * cell_size),
                 (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), thick)

    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                text = str(board[r][c])
                font_scale, thickness = 1.2, 2
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = start_x + c * cell_size + (cell_size - text_size[0]) // 2
                text_y = start_y + r * cell_size + (cell_size + text_size[1]) // 2

                # default yellow for original puzzle, else color from cell_colors
                if (r, c) in cell_colors:
                    color = cell_colors[(r, c)]
                else:
                    color = (0, 255, 255)  # preset puzzle numbers
                cv2.putText(image, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if selected_cell:
        rr, cc = selected_cell
        cv2.rectangle(image,
                      (start_x + cc * cell_size, start_y + rr * cell_size),
                      (start_x + (cc + 1) * cell_size, start_y + (rr + 1) * cell_size),
                      (0, 0, 255), 2)
    return image

def draw_ui_elements(image):
    num_pad_buttons.clear()
    for i in range(1, 9+1):
        row, col = (i - 1) // 3, (i - 1) % 3
        x, y = pad_start_x + col * (button_size + margin), pad_start_y + row * (button_size + margin)
        cv2.rectangle(image, (x, y), (x + button_size, y + button_size), (200, 200, 200), -1)
        cv2.putText(image, str(i), (x + 12, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        num_pad_buttons[i] = (x, y, x + button_size, y + button_size)

    x, y = pad_start_x + button_size + margin, pad_start_y + 3 * (button_size + margin)
    cv2.rectangle(image, (x, y), (x + button_size, y + button_size), (180, 180, 250), -1)
    cv2.putText(image, "0", (x + 12, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    num_pad_buttons[0] = (x, y, x + button_size, y + button_size)

    cv2.rectangle(image, (20, 40), (220, 90), (0, 150, 0), -1)
    cv2.putText(image, "Next Puzzle", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image

def mouse_callback(event, x, y, flags, param):
    global selected_cell, puzzle_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_x <= x <= start_x + grid_size and start_y <= y <= start_y + grid_size:
            row = (y - start_y) // cell_size
            col = (x - start_x) // cell_size
            selected_cell = (row, col)

        for num, (x1, y1, x2, y2) in num_pad_buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2 and selected_cell:
                r, c = selected_cell
                puzzles[puzzle_index][r][c] = num
                # Validate and set color
                if num == 0:
                    cell_colors.pop((r, c), None)  # remove color if clearing
                else:
                    color = (0, 255, 0) if is_valid_move(puzzles[puzzle_index], r, c, num) else (0, 0, 255)
                    cell_colors[(r, c)] = color
                break

        if 20 <= x <= 220 and 40 <= y <= 90:
            puzzle_index = (puzzle_index + 1) % len(puzzles)
            selected_cell = None
            cell_colors.clear()

def main():
    cv2.namedWindow("Sudoku Frontend")
    cv2.setMouseCallback("Sudoku Frontend", mouse_callback)
    while True:
        image = np.zeros((h, w, 3), dtype=np.uint8)
        draw_sudoku_grid(image, puzzles[puzzle_index])
        draw_ui_elements(image)
        cv2.putText(image, "Click cell + number pad | Press 'q' to Quit",
                    (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Sudoku Frontend", image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
