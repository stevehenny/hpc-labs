#include "htk.h"
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

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

int iterations;
static void hot_plate_thread(real_t *w, real_t *u, int start_row, int finish_row, int height, int width,
                             real_t epsilon, int *run)
{
  int temp_run = 0;
  for (int i = start_row; i < finish_row; ++i)  // Fix: include finish_row
  {
    if (i == 0) continue;
    if(i == height - 1) break;
    for (int j = 1; j < width - 1; ++j)
    {
      val(w, i, j) =
          (val(u, i - 1, j) + val(u, i + 1, j) + val(u, i, j - 1) + val(u, i, j + 1)) / (real_t)4;
      if (epsilon < FABS(val(w, i, j) - val(u, i, j)))
        temp_run |= 1;
    }
  }
  (*run) = temp_run;
}

static void hot_plate(real_t *out, real_t *in, int width, int height, real_t epsilon,
                      int num_threads)
{
  real_t *u = in;
  real_t *w = out;
  int run = 1;
  int numArray[num_threads];
  std::thread threads[num_threads];

  // Initialize run flags for each thread
  for (int i = 0; i < num_threads; i++)
    numArray[i] = 0;

  // Compute rows per thread
  int rows_per_thread = (height - 1) / num_threads + 1;  // Exclude boundary rows

  // Initialize output from input (need border pixels)
  memcpy(out, in, sizeof(real_t) * width * height);
  iterations = 0;

  // Iterate until the new solution differs by no more than epsilon
  while (run)
  {
    run = 0;
    int start, finish;
    for (int i = 0; i < num_threads; i++)
    {
      // Fix: Calculate start and finish, handle boundaries correctly
      start = i * rows_per_thread;
      finish = start + rows_per_thread;
    //   std::cout << "Start: " << start << std::endl;
    //   std::cout << "Finish: " << finish << std::endl;


      threads[i] = std::thread(hot_plate_thread, w, u, start, finish, height,  width, epsilon, &numArray[i]);
    }

    for (auto &th : threads)
    {
      th.join();
    }

    for (int i = 0; i < num_threads; i++)
    {
      run |= numArray[i];
    }

    // Swap pointers
    std::swap(w, u);
    iterations++;
  }

  // Copy result to output
  if (u != out)
  {
    memcpy(out, u, sizeof(real_t) * width * height);
  }
}

int main(int argc, char *argv[])
{

  int num_threads;
  {
    if (const char *str_p = std::getenv("NUM_THREADS"))
      num_threads = atoi(str_p);
    else
      num_threads = std::thread::hardware_concurrency();
  }
  htkArg_t args;
  int width;
  int height;
  int channels;
  char *inputFile;
  htkImage_t input;
  htkImage_t output;
  float *hostInputData;
  float *hostOutputData;

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

  output = htkImage_new(width, height, channels);
  hostInputData = htkImage_getData(input);
  hostOutputData = htkImage_getData(output);
  htkTime_stop(IO, "Importing data and creating memory on host");
  htkLog(TRACE, "Image dimensions WxH are ", width, " x ", height);

  htkTime_start(Compute, "Doing the computation");
  hot_plate(hostOutputData, hostInputData, width, height, EPSILON, num_threads);
  htkTime_stop(Compute, "Doing the computation");
  htkLog(TRACE, "Solution iterations: ", iterations);

  htkSolution(args, output);

  htkImage_delete(output);
  htkImage_delete(input);
  std::cout << std::endl;

  return 0;
}
