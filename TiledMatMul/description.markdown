---
title: Tiled Matrix Multiplication
lab: 4
---

## Objective

Implement a tiled dense matrix multiplication routine using shared memory.

## Prerequisites

Before starting this lab, make sure that:

- You have completed "Simple Matrix Multiplication" Lab
- You have read Kirk, PMPP Ch 4 (all)

## Instructions

Edit the code to perform the following:

- Allocate device memory
- Copy host memory to device
- Initialize thread block and kernel grid dimensions
- Invoke CUDA kernel
- Copy results from device to host
- Deallocate device memory
- Implement the matrix-matrix multiplication routine using shared memory and tiling

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

The executable generated as a result of compiling the lab can be run using the following command:

```
./solution -e <expected.raw> -i <input0.raw>,<input1.raw> -o <output.raw> -t matrix
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process. Alternatively, just type `make run`.
