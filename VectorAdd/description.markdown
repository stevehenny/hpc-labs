---
title: Vector Add
lab: 2
---

## Objective

The purpose of this lab is to introduce the student to the CUDA API by implementing vector addition. The student will implement vector addition by writing the GPU kernel code as well as the associated host code.

## Prerequisites

Before starting this lab, make sure that:

- You have completed the "Device Query" lab
- You have read Kirk, PMPP Ch 2

## Instructions

Edit the code to perform the following:

- Allocate device memory
- Copy host memory to device
- Initialize thread block and kernel grid dimensions
- Invoke CUDA kernel
- Copy results from device to host
- Free device memory
- Write the CUDA kernel

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

The executable generated as a result of compiling the lab can be run using the following command:

```
./solution -e <expected.raw> -i <intput1.raw>,<input2.raw> -o <output.raw> -t vector
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process. Alternatively, just type `make run`.
