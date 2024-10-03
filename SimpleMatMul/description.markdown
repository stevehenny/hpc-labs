---
title: Simple Matrix Multiplication
lab: 3
---

## Objective

Implement a simple dense matrix multiplication routine. Optimizations such as tiling and usage of shared memory are not required for this lab.

## Prerequisites

Before starting this lab, make sure that:

- You have completed "Vector Addition" Lab
- You have read Kirk, PMPP Ch 4.2

## Instructions

Edit the code to perform the following:

- Allocate device memory
- Copy host memory to device
- Initialize thread block and kernel grid dimensions
- Invoke CUDA kernel
- Copy results from device to host
- Deallocate device memory

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

The executable generated as a result of compiling the lab can be run using the following command:

```
./solution -e <expected.raw> -i <input0.raw>,<input1.raw> -o <output.raw> -t matrix
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process. Alternatively, just type `make run`.
