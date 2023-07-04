import matplotlib.pyplot as plt
import numpy as np

g2o_csv = "/home/luyh/myworkspace/slam_in_autonomous_driving/data/ch4/g2o.txt"
ceres_csv = "/home/luyh/myworkspace/slam_in_autonomous_driving/data/ch4/ceres.txt"

def load(filename):
    xs, ys = [], []
    with open(filename, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            x = float(values[1])
            y = float(values[2])
            xs.append(x)
            ys.append(y)
    return xs, ys

if __name__ == "__main__":
    g2o_xs, g2o_ys = load(g2o_csv)
    ceres_xs, ceres_ys = load(ceres_csv)
    plt.plot(g2o_xs, g2o_ys, "r")
    plt.plot(ceres_xs, ceres_ys, "g")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
