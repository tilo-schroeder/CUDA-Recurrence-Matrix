import torch
import numpy as np
import matplotlib.pyplot as plt

def phase_space_reconstruction(time_series, tau, m):
    N = time_series.size(0)
    reconstructed_size = N - (m - 1) * tau
    phase_space = torch.zeros((reconstructed_size, m), dtype=time_series.dtype, device=time_series.device)

    for i in range(reconstructed_size):
        for j in range(m):
            phase_space[i, j] = time_series[i + j * tau]

    return phase_space

def compute_recurrence_matrix(phase_space, epsilon):
    N, m = phase_space.shape
    recurrence_matrix = torch.zeros((N, N), dtype=torch.int32, device=phase_space.device)

    for idx in range(N):
        for idy in range(N):
            distance = torch.sum((phase_space[idx] - phase_space[idy]) ** 2)
            if torch.sqrt(distance) <= epsilon:
                recurrence_matrix[idx, idy] = 1

    return recurrence_matrix

def recurrence_matrix(signal, time_delay, embedding_dimension, epsilon):
    signal = signal.to(dtype=torch.float32)
    N = signal.size(0)

    phase_space = phase_space_reconstruction(signal, time_delay, embedding_dimension)
    recurrence_matrix_result = compute_recurrence_matrix(phase_space, epsilon)

    return recurrence_matrix_result


def main():
    t = np.linspace(0, 10*np.pi, 10000)
    values = np.sin(t)
    signal = torch.Tensor(values)
    
    y = recurrence_matrix(signal, 100, 6, 0.1)
    plt.imsave('output.png', y)

if __name__ == "__main__":
    main()