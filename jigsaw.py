import os
import time
import math
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat, loadmat


def get_or_make_lena(grid_size=(4, 4), img_size=256):
    """
    Downloads (or generates) a grayscale Lena image and saves it locally.
    """
    mat_file = 'scrambled_lena.mat'
    print("Trying to download the Lena image...")

    try:
        resp = requests.get('http://www.lenna.org/len_std.jpg', stream=True, timeout=10)
        resp.raise_for_status()
        lena_img = Image.open(resp.raw).convert('L').resize((img_size, img_size))
        lena_array = np.array(lena_img)
    except requests.exceptions.RequestException as err:
        print(f"Couldn’t download Lena image ({err}). Generating a dummy image instead.")
        lena_array = np.zeros((img_size, img_size), dtype=np.uint8)
        for x in range(img_size):
            for y in range(img_size):
                lena_array[x, y] = (x + y) % 256
    np.save('original_lena.npy', lena_array)
    print("Saved a copy as 'original_lena.npy' for future runs.")

    if os.path.exists(mat_file):
        print(f"'{mat_file}' already exists, skipping recreation.")
        return lena_array

    print("Now slicing the image into tiles and scrambling them...")
    piece_h = img_size // grid_size[0]
    piece_w = img_size // grid_size[1]

    tile_list = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            tile = lena_array[row * piece_h:(row + 1) * piece_h,
                              col * piece_w:(col + 1) * piece_w]
            tile_list.append(tile)

    random.shuffle(tile_list)
    savemat(mat_file, {
        'scrambled_pieces': np.array(tile_list, dtype=object),
        'grid_size': np.array(grid_size)
    })
    print(f"Generated and saved '{mat_file}'.")
    return lena_array


def read_scrambled_pieces(mat_file='scrambled_lena.mat'):
    """
    Loads puzzle pieces and their grid size from a .mat file.
    """
    try:
        data = loadmat(mat_file)
        grid = tuple(data['grid_size'].flatten())
        scrambled = data['scrambled_pieces']
        pieces = [np.array(scrambled[i], dtype=np.uint8).squeeze()
                  for i in range(scrambled.shape[0])]
        print(f"Loaded {len(pieces)} scrambled pieces.")
        return pieces, grid
    except Exception as err:
        print(f"Could not load pieces properly: {err}")
        return None, None


def calc_energy(layout, tiles, grid_size):
    """
    Calculates how 'unsmooth' the edges are in the current puzzle layout.
    Lower energy = better fit.
    """
    rows, cols = grid_size
    total_energy = 0.0

    for r in range(rows):
        for c in range(cols - 1):
            left_piece = tiles[layout[r, c]]
            right_piece = tiles[layout[r, c + 1]]
            total_energy += np.sum((left_piece[:, -1] - right_piece[:, 0]) ** 2)

    for r in range(rows - 1):
        for c in range(cols):
            top_piece = tiles[layout[r, c]]
            bottom_piece = tiles[layout[r + 1, c]]
            total_energy += np.sum((top_piece[-1, :] - bottom_piece[0, :]) ** 2)

    return total_energy


def solve_with_sa(tiles, grid_size, max_iters=None):
    """
    Uses simulated annealing to approximate the correct arrangement.
    """
    num_tiles = len(tiles)
    rows, cols = grid_size
    cur_layout = np.arange(num_tiles).reshape(grid_size)
    cur_energy = calc_energy(cur_layout, tiles, grid_size)
    best_layout = np.copy(cur_layout)
    best_energy = cur_energy

    T = 1e8
    Tmin = 1e-3
    alpha = 0.9999
    step = 0
    history = [(0, cur_energy)]

    print("\n=== Starting Simulated Annealing ===")
    t_start = time.time()

    while T > Tmin:
        if max_iters and step >= max_iters:
            break

        for _ in range(rows * cols):
            step += 1
            r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
            r2, c2 = random.randint(0, rows - 1), random.randint(0, cols - 1)

            new_layout = np.copy(cur_layout)
            new_layout[r1, c1], new_layout[r2, c2] = new_layout[r2, c2], new_layout[r1, c1]

            new_energy = calc_energy(new_layout, tiles, grid_size)
            diff = new_energy - cur_energy

            if diff < 0 or random.random() < math.exp(-diff / T):
                cur_layout = new_layout
                cur_energy = new_energy
                if cur_energy < best_energy:
                    best_layout = np.copy(cur_layout)
                    best_energy = cur_energy
                    history.append((step, best_energy))

        T *= alpha
        if step % 100 == 0:
            print(f"Iter {step:6d} | Temp {T:10.2e} | Best Energy {best_energy:12.2f}")

    print(f"\n=== Annealing Complete ({time.time() - t_start:.2f}s) ===")
    print(f"Initial Energy: {history[0][1]:.2f}")
    print(f"Final Energy:   {best_energy:.2f}")
    return best_layout, history


def rebuild_image(layout, tiles, grid_size):
    """
    Reconstructs the full image from given tiles and arrangement.
    """
    rows, cols = grid_size
    ph, pw = tiles[0].shape
    final_img = np.zeros((rows * ph, cols * pw), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            final_img[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = tiles[layout[r, c]]
    return final_img


def show_results(tiles, final_layout, grid_size, history=None, original=None):
    """
    Displays original, scrambled, and solved puzzles side-by-side.
    """
    start_layout = np.arange(len(tiles)).reshape(grid_size)
    scrambled_img = rebuild_image(start_layout, tiles, grid_size)
    solved_img = rebuild_image(final_layout, tiles, grid_size)

    fig, axes = plt.subplots(1, 4 if history else 3, figsize=(20, 6))

    if original is not None:
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    else:
        axes[0].imshow(scrambled_img, cmap='gray')
        axes[0].set_title("Original (Unavailable)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(scrambled_img, cmap='gray')
    axes[1].set_title("Initial Scramble", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(solved_img, cmap='gray')
    axes[2].set_title("Solved Puzzle (Simulated Annealing)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    if history:
        ax_hist = axes[3]
        iters, energies = zip(*history)
        ax_hist.plot(iters, energies, 'b-', linewidth=2)
        ax_hist.set_yscale('log')
        ax_hist.set_xlabel('Iterations')
        ax_hist.set_ylabel('Energy')
        ax_hist.set_title("Energy Convergence", fontsize=14, fontweight='bold')
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    original_img = get_or_make_lena(grid_size=(4, 4), img_size=256)
    tiles, grid_sz = read_scrambled_pieces()

    if tiles:
        best_layout, hist = solve_with_sa(tiles, grid_sz, max_iters=250000)
        show_results(tiles, best_layout, grid_sz, hist, original_img)
    else:
        print("No puzzle pieces were loaded successfully.")

