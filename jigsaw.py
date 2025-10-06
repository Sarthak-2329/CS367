import numpy as np
import matplotlib.pyplot as plt
import random
import math

def create_scrambled_lena():
    """
    Downloads the Lena image, converts to grayscale, centers and crops,
    then divides it into patches and shuffles them to create a scrambled version.
    Returns the scrambled image.
    """
    try:
        # Download Lena image and convert to grayscale
        image = plt.imread('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png')
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) * 255
        gray = gray.astype(np.uint8)

        # Crop to 512x512 centered region
        h, w = gray.shape
        top = (h - 512) // 2
        left = (w - 512) // 2
        cropped = gray[top:top+512, left:left+512]

        # Divide into 16 (4x4) patches and randomly permute them
        patch_size = 128
        dim = 4
        patches = []
        positions = list(range(dim * dim))
        for r in range(dim):
            for c in range(dim):
                patches.append(cropped[r*patch_size:(r+1)*patch_size,
                                       c*patch_size:(c+1)*patch_size])
        np.random.shuffle(positions)

        # Fill scrambled image with shuffled patches
        scrambled = np.zeros_like(cropped)
        idx = 0
        for r in range(dim):
            for c in range(dim):
                scrambled[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = patches[positions[idx]]
                idx += 1
        return scrambled
    except Exception as e:
        print(f"Could not process Lena image: {e}")
        print("Using random image.")
        return np.random.randint(0, 256, (512, 512), dtype=np.uint8)

def get_patches():
    """Gets scrambled Lena image, slices into 4x4 grid of square patches."""
    scrambled = create_scrambled_lena()
    patch_size = 128
    dim = 4
    patches = []
    for r in range(dim):
        for c in range(dim):
            patches.append(scrambled[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size])
    return patches, patch_size, dim, scrambled

def edge_difference(patch1, patch2, direction='H'):
    """Returns squared difference of right edge of patch1 and left of patch2 (if 'H'), or bottom/top (if 'V')."""
    if direction == 'H':
        side1 = patch1[:, -1]
        side2 = patch2[:, 0]
    else:
        side1 = patch1[-1, :]
        side2 = patch2[0, :]
    return np.sum((side1.astype('float') - side2.astype('float'))**2)

def total_grid_score(grid, patches):
    """Sum of edge differences for every neighboring patch pair in the given grid."""
    score = 0
    dim = grid.shape[0]
    for r in range(dim):
        for c in range(dim):
            idx = grid[r, c]
            if c < dim - 1:
                right_idx = grid[r, c+1]
                score += edge_difference(patches[idx], patches[right_idx], 'H')
            if r < dim - 1:
                down_idx = grid[r+1, c]
                score += edge_difference(patches[idx], patches[down_idx], 'V')
    return score

def anneal_jigsaw(patches, dim, start_temp, cool, steps):
    """Optimizes patch arrangement using simulated annealing to minimize total_grid_score."""
    N = len(patches)
    current = np.arange(N).reshape((dim, dim))
    np.random.shuffle(current.flat)
    cur_score = total_grid_score(current, patches)
    best = np.copy(current)
    best_score = cur_score
    temp = start_temp

    print(f"Initial Score: {cur_score:.2f}")
    for i in range(steps):
        trial = np.copy(current)
        r1, c1, r2, c2 = [random.randint(0, dim-1) for _ in range(4)]
        trial[r1, c1], trial[r2, c2] = trial[r2, c2], trial[r1, c1]
        trial_score = total_grid_score(trial, patches)
        diff = trial_score - cur_score
        if diff < 0 or (temp > 0 and random.random() < math.exp(-diff / temp)):
            current = trial
            cur_score = trial_score
        if cur_score < best_score:
            best = np.copy(current)
            best_score = cur_score
        temp *= cool
        if i % 100 == 0:
            print(f"Iteration {i}/{steps} | Temp: {temp:.2f} | Current: {cur_score:.2f} | Best: {best_score:.2f}")
    return best, best_score

def build_image(grid, patches, patch_size):
    """Constructs the image from a grid of patch indices."""
    dim = grid.shape[0]
    size = dim * patch_size
    output = np.zeros((size, size), dtype=np.uint8)
    for r in range(dim):
        for c in range(dim):
            idx = grid[r, c]
            output[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = patches[idx]
    return output

if __name__ == '__main__':
    try:
        patches, patch_size, dim, scrambled = get_patches()
    except Exception as e:
        print(f"Error preparing image: {e}")
        exit()

    INITIAL_TEMP = 100000
    COOLING = 0.999
    STEPS = 5000

    print("--- Simulated Annealing Puzzle Solver ---")
    solved_grid, score = anneal_jigsaw(patches, dim, INITIAL_TEMP, COOLING, STEPS)

    print("\n--- Done ---")
    print(f"Best Score: {score:.2f}")
    print("Grid:\n", solved_grid)

    solved_img = build_image(solved_grid, patches, patch_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(scrambled, cmap='gray')
    axes[0].set_title('Scrambled Image')
    axes[0].axis('off')
    axes[1].imshow(solved_img, cmap='gray')
    axes[1].set_title(f'Reconstructed (Score: {score:.2f})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
