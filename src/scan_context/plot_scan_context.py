import matplotlib.pyplot as plt
import numpy as np
import sys

def load_as_np(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        shape_values = lines[0].strip().split(" ")
        rows, cols = int(shape_values[0]), int(shape_values[1])
        data_values = lines[1].strip().split(" ")
        data = list(map(float, data_values))
        np_data = np.array(data).reshape((rows, cols))
        return np_data
    
def plot_as_heatmap(data):
    plt.imshow(data, cmap='coolwarm')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    filename = "./data/scan_context/first.sc.txt"
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    
    data = load_as_np(filename)
    plot_as_heatmap(data)