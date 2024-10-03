
#include "htk.h"

#define htkCheck(stmt)                                                \
  do {                                                                \
    cudaError_t err = stmt;                                           \
    if (err != cudaSuccess) {                                         \
      htkLog(ERROR, "Failed to run stmt ", #stmt);                    \
      htkLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

// Compute C = A * B
// sgemm stands for single precision general matrix-matrix multiply
__global__ void sgemm(float *A, float *B, float *C,
                      int numARows, int numAColumns,
                      int numBRows, int numBColumns,
                      int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;
  if((Row < numCRows) && (Col < numCColumns)){
    float Pvalue = 0;
    for (int i = 0; i < numAColumns; i++){
        Pvalue += A[i+Row*numAColumns] * B[i*numBColumns+Col];
    }
    C[Row*numCColumns+Col] = Pvalue;
  }
												 

}

int main(int argc, char **argv) {
  htkArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C
  int numCColumns; // number of columns in the matrix C

  args = htkArg_read(argc, argv);
  if (args.inputCount != 2) {htkLog(ERROR, "Missing input"); return 1;}

  htkTime_start(IO, "Importing data and creating memory on host");
  hostA = (float *)htkImport(htkArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *)htkImport(htkArg_getInputFile(args, 1), &numBRows, &numBColumns);
  numCRows    = numARows;
  numCColumns = numBColumns;
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  htkTime_stop(IO, "Importing data and creating memory on host");
  htkLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  htkLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  htkLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  htkTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int Asize = sizeof(float) * numAColumns * numARows;
  int Bsize = sizeof(float) * numBColumns * numBRows;
  int Csize = sizeof(float) * numCColumns * numCRows;
  cudaMalloc((void **)&deviceA, Asize);
  cudaMalloc((void **)&deviceB, Bsize);
  cudaMalloc((void **)&deviceC, Csize);

  htkTime_stop(GPU, "Allocating GPU memory.");

  htkTime_start(Copy, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, Bsize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, Csize, cudaMemcpyHostToDevice);

  htkTime_stop(Copy, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/16.0), ceil(numCRows / 16.0), 1);
  dim3 DimBlock(16, 16, 1);

  htkTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  sgemm<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows,
  numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  htkTime_stop(Compute, "Performing CUDA computation");

  htkTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, Csize, cudaMemcpyDeviceToHost);

  htkTime_stop(Copy, "Copying output memory to the CPU");

  htkTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  htkTime_stop(GPU, "Freeing GPU Memory");

  htkSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
