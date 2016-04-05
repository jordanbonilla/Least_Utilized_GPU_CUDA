//--------------------------------------------------------------------------
// Project: 
// Select the least utilized GPU on a CUDA-enabled system.
// Insert into your code as desired. 
//
// Prerequisites: 
// Must have installed the CUDA toolkit.
// Must be running on a UNIX machine
//
// Independent testing info:
// Compile on commandline: nvcc least_utilized_GPU.cu -o test
// run on commandline: ./test
//
// Author: Jordan Bonilla
// Date  : April 2016
// License: All rights Reserved. See LICENSE.txt
//--------------------------------------------------------------------------

#include <cstdio> // printf
#include <stdlib.h> // popen, pclose, atoi, fread
#include <cuda_runtime.h> // cudaGetDeviceCount, cudaSetDevice

// Select the least utilized GPU on this system. Estimate
// GPU utilization using GPU temperature. UNIX only.
void select_GPU() 
{
  // Get the number of GPUs on this machine
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if(num_devices == 1) {
    return;
  }
  // Read GPU info into buffer "output"
  const unsigned int MAX_BYTES = 10000;
  char output[MAX_BYTES];
  FILE *fp = popen("nvidia-smi &> /dev/null", "r");
  fread(output, sizeof(char), MAX_BYTES, fp);
  pclose(fp);
  // array to hold GPU temperatures
  int * temperatures = new int[num_devices];
  // parse output for temperatures using knowledge of "nvidia-smi" output format
  int i = 0;
  unsigned int num_temps_parsed = 0;
  while(output[i] != '\0') {
    if(output[i] == '%') {
      unsigned int temp_begin = i + 1;
      while(output[i] != 'C') {
        ++i;
      }
      unsigned int temp_end = i;
      char this_temperature[32];
      // Read in the characters cooresponding to this temperature
      for(int j = 0; j < temp_end - temp_begin; ++j) {
          this_temperature[j] = output[temp_begin + j];
      }
      this_temperature[temp_end - temp_begin + 1] = '\0';
      // Convert the string representation to an int
      temperatures[num_temps_parsed] = atoi(this_temperature);
      num_temps_parsed++;
    }
    ++i;
  }
  // Get GPU with lowest temperature
  int min_temp = 1e7, index_of_min = -1;
  for (int i = 0; i < num_devices; i++) 
  {
    int candidate_min = temperatures[i];
    if(candidate_min < min_temp) 
    {
      min_temp = candidate_min;
      index_of_min = i;
    }
  }
  // Tell CUDA to use the GPU with the lowest temeprature
  printf("Index of the GPU with the lowest temperature: %d (%d C)\n", 
    index_of_min, min_temp);
  cudaSetDevice(index_of_min);
  // Free memory and return
  delete(temperatures);
  return;
}

int main(int argc, char **argv) {
  select_GPU();
  return 0;
}


