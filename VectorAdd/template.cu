#include "htk.h"

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
}

int main(int argc, char **argv) {
  htkArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = htkArg_read(argc, argv);
  if (args.inputCount != 2) {htkLog(ERROR, "Missing input"); return 1;}

  htkTime_start(IO, "Importing data and creating memory on host");
  hostInput1 = (float *)htkImport(htkArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)htkImport(htkArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  htkTime_stop(IO, "Importing data and creating memory on host");

  htkLog(TRACE, "The input length is ", inputLength);

  htkTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  htkTime_stop(GPU, "Allocating GPU memory.");

  htkTime_start(Copy, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  htkTime_stop(Copy, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  htkTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  cudaDeviceSynchronize();
  htkTime_stop(Compute, "Performing CUDA computation");

  htkTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  htkTime_stop(Copy, "Copying output memory to the CPU");

  htkTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  htkTime_stop(GPU, "Freeing GPU Memory");

  htkSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
