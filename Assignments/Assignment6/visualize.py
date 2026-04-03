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
        binary = np.fromfile(filename, dtype = np.int32).reshape((N, N))
        out_png = filename.replace(".raw", ".png")
        plt.figure(figsize=(6, 6))
        plt.imshow(binary, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi = 150)
        plt.close()
        alive = binary.sum()
        print(f"{filename} -> {out_png} | Alive Cells: {alive}")
    except Exception as e:
        print (f"Error processing {filename}: {e}")