#include "htk.h"
#include <npp.h>

#define htkCheck(stmt)                                                \
  do {                                                                \
    cudaError_t err = stmt;                                           \
    if (err != cudaSuccess) {                                         \
      htkLog(ERROR, "Failed to run stmt ", #stmt);                    \
      htkLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

#define nppCheck(stmt)                                                \
  do {                                                                \
    NppStatus err = stmt;                                           \
    if (err != NPP_SUCCESS) {                                         \
      htkLog(ERROR, "Failed to run stmt ", #stmt);                    \
      htkLog(ERROR, "Got CUDA error ...  ", (err)); \
      exit(1);                                                        \
    }                                                                 \
  } while (0)


#if defined(USE_DOUBLE)
#define EPSILON 0.00005
#define FABS fabs
typedef double real_t;

#else
#define EPSILON 0.00005f
#define FABS fabsf
typedef float real_t;
#endif
#define val(arry, i, j) arry[(i)*width + (j)]

int iterations;

static void hot_plate(real_t *out, real_t *in, int width, int height, real_t epsilon) {
    htkCheck(cudaMemcpy(out, in, width * height * sizeof(real_t), cudaMemcpyDeviceToDevice));
    real_t *u_offset = in + width + 1;
    real_t *w_offset = out + width + 1;
    int iter = 0;
    Npp8u *d_run;
    htkCheck(cudaMalloc((void**)&d_run, sizeof(Npp8u)));

    int stride;
    Npp8u *runBuffer = nppiMalloc_8u_C1(width-2, height-2, &stride);

    Npp32s lineStep = width * sizeof(real_t);
    NppiSize oSrcSize = { .width=width, .height=height };
    NppiPoint oSrcOffset = { .x=1, .y=1 };
    NppiSize RegionOfIntrest = { width - 2, height - 2 };

    size_t h_buffer_size;
    nppCheck(nppiMinGetBufferHostSize_8u_C1R(RegionOfIntrest, &h_buffer_size));
    Npp8u *scratchBuffer;
    htkCheck(cudaMalloc((void **)&scratchBuffer, h_buffer_size));

    const Npp32f pKernelHost[9] = { 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0 };
    Npp32f *pKernelDevice;
    htkCheck(cudaMalloc((void**)&pKernelDevice, sizeof(pKernelHost)));
    htkCheck(cudaMemcpy(pKernelDevice, pKernelHost, sizeof(pKernelHost), cudaMemcpyHostToDevice));

    NppiSize kernelSize = { 3, 3 };
    NppiPoint oAnchor = { 1, 1 };
    Npp8u h_run = 0;


    while (!h_run) {
        nppCheck(nppiFilterBorder_32f_C1R(u_offset, lineStep, oSrcSize, oSrcOffset,
                                                   w_offset, lineStep, RegionOfIntrest,
                                                   pKernelDevice, kernelSize, oAnchor, NPP_BORDER_REPLICATE));

        
        nppCheck(nppiCompareEqualEps_32f_C1R(u_offset, lineStep, w_offset, lineStep, runBuffer, stride, RegionOfIntrest, (Npp32f)EPSILON));

        nppCheck(nppiMin_8u_C1R(runBuffer, stride, RegionOfIntrest, scratchBuffer, d_run));

        // Swap buffers
        { real_t *t = w_offset; w_offset = u_offset; u_offset = t; }

        iter++;
        htkCheck(cudaMemcpy(&h_run, d_run, sizeof(Npp8u), cudaMemcpyDeviceToHost));
    }

    if (u_offset != out + width + 1) {
        htkCheck(cudaMemcpy(out, in, sizeof(real_t) * width * height, cudaMemcpyDeviceToDevice));
    }

    iterations = iter;
    
    nppiFree(runBuffer);
    htkCheck(cudaFree(scratchBuffer));
    htkCheck(cudaFree(pKernelDevice));
    htkCheck(cudaFree(d_run));
}


int main(int argc, char *argv[]) {
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
 
  args = htkArg_read(argc, argv);
  if (args.inputCount != 1) {htkLog(ERROR, "Missing input"); return 1;}

  htkTime_start(IO, "Importing data and creating memory on host");
  inputFile = htkArg_getInputFile(args, 0);
  input = htkImport(inputFile);
  width  = htkImage_getWidth(input);
  height = htkImage_getHeight(input);
  channels  = htkImage_getChannels(input);
  if (channels != 1) {htkLog(ERROR, "Expecting gray scale image"); return 1;}
  output = htkImage_new(width, height, channels);
  hostInputData  = htkImage_getData(input);
  hostOutputData = htkImage_getData(output);
  htkTime_stop(IO, "Importing data and creating memory on host");
  htkLog(TRACE, "Image dimensions WxH are ", width, " x ", height);

// CUDA MALLOC
int size = width * height * sizeof(float);
htkCheck(cudaMalloc((void **)&deviceInputData, size));

htkCheck(cudaMalloc((void **)&deviceOutputData, size));

// CUDA COPY
htkCheck(cudaMemcpy(deviceInputData, hostInputData, size, cudaMemcpyHostToDevice));





  htkTime_start(Compute, "Doing the computation");
  hot_plate(deviceOutputData, deviceInputData, width, height, EPSILON);
  htkTime_stop(Compute, "Doing the computation");
  htkLog(TRACE, "Solution iterations: ", iterations);

htkCheck(cudaMemcpy(hostOutputData, deviceOutputData, size, cudaMemcpyDeviceToHost));
  htkSolution(args, output);

  htkImage_delete(output);
  htkImage_delete(input);

htkCheck(cudaFree(deviceInputData));
htkCheck(cudaFree(deviceOutputData));

  return 0;
}
