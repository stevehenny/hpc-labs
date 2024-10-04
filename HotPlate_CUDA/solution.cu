#include "htk.h"


#if defined(USE_DOUBLE)
#define EPSILON 0.00005
#define FABS fabs
typedef double real_t;

#else
#define EPSILON 0.00005f
#define FABS fabsf
typedef float real_t;
#endif

#define val(arry, i, j) arry[(i) * width + (j)]
#define BLOCK_SIZE 16
// int iterations;
__device__ int d_run;

__global__ static void hot_plate_kernel(real_t *out, real_t *in, int width, int height,
                                        real_t epsilon)
{

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int r_run = 0;
  if ((Row < height - 1) && (Col < width - 1) && (Row > 0) && (Col > 0)){
    out[Row * width + Col] =
    (in[(Row-1) * width + Col] + in[(Row+1)*width + Col] + in[(Row * width) + Col + 1] + in[(Row * width) + Col - 1]) / (real_t)4;
  if (epsilon < FABS(out[Row * width + Col] - in[Row * width + Col])){
    r_run |= 1;
    
    
}
}
r_run = __syncthreads_or(r_run);
if(threadIdx.x == 0 && threadIdx.y == 0)
atomicOr(&d_run, r_run);
}


int main(int argc, char *argv[])
{
  htkArg_t args;
  int width;
  int height;
  int channels;
  char *inputFile;
  htkImage_t input;
  htkImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;
  int h_run = 1;

  args = htkArg_read(argc, argv);
  if (args.inputCount != 1)
  {
    htkLog(ERROR, "Missing input");
    return 1;
  }

  htkTime_start(IO, "Importing data and creating memory on host");
  inputFile = htkArg_getInputFile(args, 0);
  input = htkImport(inputFile);
  width = htkImage_getWidth(input);
  height = htkImage_getHeight(input);
  channels = htkImage_getChannels(input);
  if (channels != 1)
  {
    htkLog(ERROR, "Expecting gray scale image");
    return 1;
  }
  int size = sizeof(float) * width * height;

  // CUDA MALLOC
  cudaMalloc((void **)&deviceInputData, size);
  cudaMalloc((void **)&deviceOutputData, size);


  output = htkImage_new(width, height, channels);
  hostInputData = htkImage_getData(input);
  hostOutputData = htkImage_getData(output);
  htkTime_stop(IO, "Importing data and creating memory on host");
  htkLog(TRACE, "Image dimensions WxH are ", width, " x ", height);

  // CUDA COPY
  cudaMemcpy(deviceInputData, hostInputData, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutputData, hostInputData, size, cudaMemcpyHostToDevice);


  dim3 DimGrid(ceil(width / (float)BLOCK_SIZE), ceil(height / (float)BLOCK_SIZE), 1);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  
  real_t *u = deviceInputData;
  real_t *w = deviceOutputData;
  int iterations = 0;

  htkTime_start(Compute, "Doing the computation");
  while(h_run){
    h_run = 0;
    
    cudaMemcpyToSymbol(d_run, &h_run, sizeof(int));
    hot_plate_kernel<<<DimGrid, DimBlock>>>(w, u, width, height,
      EPSILON);
      cudaDeviceSynchronize();
      cudaMemcpyFromSymbol(&h_run, d_run, sizeof(int));
      {
        real_t* t = w;
        w = u;
        u = t;
      }
      iterations++;
  }
  htkTime_stop(Compute, "Doing the computation");
  htkLog(TRACE, "Solution iterations: ", iterations);
  if (u != deviceOutputData){
    cudaMemcpy(deviceOutputData, u, size, cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(hostOutputData, deviceOutputData, size, cudaMemcpyDeviceToHost);
  htkSolution(args, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  htkImage_delete(output);
  htkImage_delete(input);

  return 0;
}
