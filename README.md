# CUDA-Recurrence-Plot
A simple CUDA kernel implementation to compute a recurrence matrix using PyTorch

Using the GPU to compute Recurrence Plots seems to offer a significant speed up over a naive Python implementation. These are just some example data points I measured:


|Number of data points|CUDA|Naive|
|---------------------|----|-----|
|100|0.0003 s|0.08 s|
|1000|0.0003 s|7.51 s|
|10000|0.02 s|2336 s|