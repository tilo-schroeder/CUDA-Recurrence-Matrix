#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void phaseSpaceReconstruction(float *timeSeries, float *phaseSpace, int N, int tau, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N - (m - 1) * tau; i += stride)
    {
        for (int j = 0; j < m; ++j)
        {
            phaseSpace[i * m + j] = timeSeries[i + j * tau];
        }
    }
}

__global__ void computeRecurrenceMatrix(float *phaseSpace, int *recurrenceMatrix, float epsilon, int N, int m)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx >= N || idy >= N)
        return;

    float distance = 0.0;
    for (int d = 0; d < m; ++d)
    {
        float diff = phaseSpace[idx * m + d] - phaseSpace[idy * m + d];
        distance += diff * diff;
    }

    if (sqrt(distance) <= epsilon)
    {
        recurrenceMatrix[idx * N + idy] = 1;
    }
    else
    {
        recurrenceMatrix[idx * N + idy] = 0;
    }
}

torch::Tensor recurrence_matrix(torch::Tensor signal, int time_delay, int embedding_dimension, float epsilon)
{
    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
    signal = signal.to(options);

    auto N = signal.size(0);

    // The reconstructed phase space will have (N - (embedding_dimension - 1) * time_delay) rows and embedding_dimension columns
    int reconstructedSize = N - (embedding_dimension - 1) * time_delay;
    auto phaseSpace = torch::empty({reconstructedSize, embedding_dimension}, options);
    auto recurrenceMatrix = torch::empty({reconstructedSize, reconstructedSize}, options.dtype(torch::kInt32));

    // Extract raw pointers to pass to CUDA kernels
    float *signal_ptr = signal.data_ptr<float>();
    float *phase_space_ptr = phaseSpace.data_ptr<float>();
    int *recurrence_matrix_ptr = recurrenceMatrix.data_ptr<int>();

    // Define CUDA kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    phaseSpaceReconstruction<<<gridSize, blockSize>>>(signal_ptr, phase_space_ptr, N, time_delay, embedding_dimension);

    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Adjust grid size for the recurrence matrix computation if necessary
    gridSize = dim3((reconstructedSize + blockSize.x - 1) / blockSize.x, (reconstructedSize + blockSize.y - 1) / blockSize.y);

    computeRecurrenceMatrix<<<gridSize, blockSize>>>(phase_space_ptr, recurrence_matrix_ptr, epsilon, reconstructedSize, embedding_dimension);

    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return recurrenceMatrix;
}