import cv2
import numpy as np

#  Sudoku Puzzles (only a few for demo, add more if needed) 
puzzles = [
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
    [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]
    ]
]


def draw_sudoku_grid(image, board):
    """Draws Sudoku grid and numbers"""
    h, w, _ = image.shape
    grid_size = min(h, w) - 100
    start_x = (w - grid_size) // 2
    start_y = (h - grid_size) // 2
    cell_size = grid_size // 9

    # Draw grid lines
    for i in range(10):
        thick = 3 if i % 3 == 0 else 1
        cv2.line(image, (start_x + i * cell_size, start_y),
                 (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), thick)
        cv2.line(image, (start_x, start_y + i * cell_size),
                 (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), thick)

    # Draw numbers
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                text = str(board[i][j])
                font_scale = 1.2
                thickness = 2
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = start_x + j * cell_size + (cell_size - text_size[0]) // 2
                text_y = start_y + i * cell_size + (cell_size + text_size[1]) // 2
                cv2.putText(image, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

    return image

def draw_ui_elements(image):
    """Draw number pad + next puzzle button"""
    # Number Pad
    button_size = 50
    margin = 15
    pad_start_x = image.shape[1] - (button_size * 3) - (margin * 4)
    pad_start_y = (image.shape[0] - (button_size * 4) - (margin * 3)) // 2

    for i in range(1, 10):
        row, col = (i - 1) // 3, (i - 1) % 3
        x, y = pad_start_x + col * (button_size + margin), pad_start_y + row * (button_size + margin)
        cv2.rectangle(image, (x, y), (x + button_size, y + button_size), (200, 200, 200), -1)
        cv2.putText(image, str(i), (x + 12, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Zero Button
    x, y = pad_start_x + button_size + margin, pad_start_y + 3 * (button_size + margin)
    cv2.rectangle(image, (x, y), (x + button_size, y + button_size), (180, 180, 250), -1)
    cv2.putText(image, "0", (x + 12, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Next Puzzle Button
    cv2.rectangle(image, (20, 40), (220, 90), (0, 150, 0), -1)
    cv2.putText(image, "Next Puzzle", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image

# --- Main Loop 
def main():
    puzzle_index = 0
    while True:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Draw Sudoku Grid
        draw_sudoku_grid(image, puzzles[puzzle_index])

        # Draw UI
        draw_ui_elements(image)

        # Instructions
        cv2.putText(image, "Press 'n' for Next Puzzle, 'q' to Quit",
                    (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Sudoku Frontend", image)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('n'):
            puzzle_index = (puzzle_index + 1) % len(puzzles)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
