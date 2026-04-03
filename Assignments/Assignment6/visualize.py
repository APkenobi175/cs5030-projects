import numpy as np
import matplotlib.pyplot as plt

N=1024
files = {
    "cpu_output.raw": "CPU Output",
    "gpu_naive_output.raw": "GPU Naive Output",
    "gpu_tiled_output.raw": "GPU Tiled Output"
}

for filename, title in files.items():
    try:
        binary = np.fromfile(filename, dtype=np.int32).reshape((N, N))
        out_png = filename.replace(".raw", ".png")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(binary, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(out_png, dpi=150, facecolor='black')
        plt.close()
        alive = binary.sum()
        print(f"{filename} -> {out_png} | Alive Cells: {alive}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")