#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitap.cuh"

__device__ long long int GetInt (char * a, int len, long long int * value) {	
	for (int i = 0; i < len; ++i) {
		char c = a[i];
		int term = 0;
		value[0] <<= 1;
		value[1] <<= 1;
		value[2] <<= 1;
		value[3] <<= 1;
		value[term] |= 1;
	}
	return 1 << (len - 1);
}

__device__ long long int BitShift(int value) {
	return (value << 1) | 1;
}

__device__ int BitapProcess (char* a, int aLen, char *b, int bLen) {
	long long int mask[4] = {0, 0 , 0, 0};

	//long long int aLimit = GetInt(a, aLen, mask);
	
	for (int i = 0; i < aLen; ++i) {
		mask[a[i]] |= (1 << i);
	}

	long long int res[MAX_ERROR];
	for (int i = 0; i < MAX_ERROR; ++i) {
		res[i] = 0;
	}

	for (int i = 0; i < bLen; ++i) {
		long long int lastValue = res[0];
		res[0] = BitShift(res[0]);
		res[0] &= mask[b[i]];
		for (int t = 1; t < MAX_ERROR; ++t) {
			long long int insert = lastValue;
			long long int del = BitShift(res[t - 1]);
			long long int replace = BitShift(lastValue);
			lastValue = res[t];
			res[t] = BitShift(res[t]);
			res[t] = (res[t] & mask[b[i]]) | insert | del | replace;
		}
		for (int t = 0; t < MAX_ERROR; ++t) {
			//значение eps 0.125
			if (i + 1>= MIN_LEN_FOR_SAMPLE && t <  0.25 * (i + 1) && (res[t] & (1 << i))) {
				return i + 1;
			}
		}	
	}
	//todo
	//если не нашли ничего подходящего пока возвращаем 0
	return 0;
}

__global__ void bitap (char * a, char * b, int * aPos, int * bPos, int* data, int * ans, int count) {
	int bx = threadIdx.x;
	int y = blockIdx.x;
	int x = bx + y * BLOCK_SIZE;
	__shared__ char parts [BLOCK_SIZE][4 * 64];
	if (x >= count) return;
	
	int aN = aPos[data[4 * x]];
	int aP = data[4 * x + 2];
	int aLen = aPos[data[4 * x] + 1] - aN;

	int bN = bPos[data[4 * x + 1]];
	int bP = data[4 * x + 3];
	int bLen = bPos[data[4 * x] + 1] - bN;

	//копируем подстроки. по 64 символа (ну или меньше) с каждой стороны от k-мера
	int aLenRev;
	if (aP >= 64) {
		aLenRev = 64;
	} else {
		aLenRev = aP;
	}
	for (int i = 0; i < aLenRev; ++i) {
		parts[bx][i] = a[aN + aP - i - 1];
	}

	int aLenFor;
	if (aP + K_MER_LEN + 64>= aLen) {
		if (aLen - aP - K_MER_LEN > 0)
			aLenFor = aLen - aP - K_MER_LEN;
		else aLenFor = 0;
	} else {
		aLenFor = 64;
	}
	for (int i = 0; i < aLenFor; ++i) {
		parts[bx][64 + i] = a[aN + aP + K_MER_LEN + i + 1];
	}

	int bLenRev;
	if (bP >= 64) {
		bLenRev = 64;
	} else {
		bLenRev = bP;
	}
	for (int i = 0; i < bLenRev; ++i) {
		parts[bx][64 * 2 + i] = b[bN + bP - i - 1];
	}

	int bLenFor;
	if (bP + K_MER_LEN + 64 >= bLen) {
		if (bLen - bP - K_MER_LEN > 0)
			bLenFor = bLen - bP - K_MER_LEN;
		else bLenFor = 0;
	} else {
		bLenFor = 64;
	}
	for (int i = 0; i < bLenFor; ++i) {
		parts[bx][64 * 3 + i] = b[bN + bP + K_MER_LEN + i + 1];
	}
	//готово

	
	int left = BitapProcess(parts[bx], aLenRev, parts[bx] + 2 * 64, bLenRev);
	int right = BitapProcess(parts[bx] + 64, aLenFor, parts[bx] + 64 * 3, bLenFor);

	if (left != 0 || right != 0) {
		ans[6 * x] = data[4 * x];
		ans[6 * x + 1] = data[4 * x + 1];
		ans[6 * x + 2] = aP - left;
		ans[6 * x + 3] = aP + K_MER_LEN + right;
		ans[6 * x + 4] = bP - left;
		ans[6 * x + 5] = bP + K_MER_LEN + right;
	} else {
		ans[6 * x] = -1;
	}
} 

void Bitap (dim3 blocks, dim3 threads,char * a, char * b, int * aPos, int * bPos, int* data, int *ans, int count) {
	bitap <<<blocks, threads>>> (a, b, aPos, bPos, data, ans, count);
}