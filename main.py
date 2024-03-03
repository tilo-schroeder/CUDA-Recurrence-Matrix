from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt


def compile_extension(name, functions):
    cuda_src = Path("rp.cu").read_text()
    cpp_src = "torch::Tensor recurrence_matrix(torch::Tensor signal, int time_delay, int embedding_dimension, float epsilon);"

    # Load the CUDA kernel as a PyTorch extension
    rp_extension = load_inline(
        name=name,
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=functions,
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rp_extension

def main():
    ext = compile_extension("recurrence_matrix_extension", ["recurrence_matrix"])

    t = np.linspace(0, 10*np.pi, 10000)
    values = np.sin(t)
    signal = torch.Tensor(values).cuda()

    y = ext.recurrence_matrix(signal, 100, 6, 0.1)

    plt.imsave('output.png', y.cpu())

if __name__ == "__main__":
    main()