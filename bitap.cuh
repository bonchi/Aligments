#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int MAX_ERROR = 10;
const int MIN_LEN_FOR_SAMPLE = 10;
const int BLOCK_SIZE=16;
const int K_MER_LEN = 14;


void Bitap (dim3, dim3, char *, char *, int *, int *, int*, int*, int);