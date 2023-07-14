import matplotlib.pyplot as plt
import numpy as np

def load_scores(filename):
    scores = []
    with open(filename, "r") as f:
        for line in f:
            s = float(line.strip())
            scores.append(s)
    return scores

if __name__ == "__main__":
    homebrew_ndt_scores = load_scores("./data/ch9/homebrew_ndt_score.txt")
    pcl_ndt_scores = load_scores("./data/ch9/pcl_ndt_score.txt")
    plt.subplot(2, 1, 1)
    plt.plot(range(len(homebrew_ndt_scores)), homebrew_ndt_scores, "r.")
    plt.grid(True)
    homebrew_ndt_scores = np.array(homebrew_ndt_scores)
    homebrew_mean = np.mean(homebrew_ndt_scores)
    homebrew_stddev = np.std(homebrew_ndt_scores)
    plt.title(f"homebrew, mean: {homebrew_mean:.3f}, stddev: {homebrew_stddev:.3f}")

    plt.subplot(2, 1, 2)
    plt.plot(range(len(pcl_ndt_scores)), pcl_ndt_scores, "g.")
    plt.grid(True)
    pcl_ndt_scores = np.array(pcl_ndt_scores)
    pcl_mean = np.mean(pcl_ndt_scores)
    pcl_stddev = np.std(pcl_ndt_scores)
    plt.title(f"pcl, mean: {pcl_mean:.3f}, stddev: {pcl_stddev:.3f}")

    plt.show()