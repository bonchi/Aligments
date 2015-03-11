#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

const int K_MER_LEN = 14;
const int HASH_COEFF = 5;

const int DIAG_COUNT = 32;
const int COMMON_PLACE = 5;

const int MAX_ERROR = 10;
const int MIN_LEN_FOR_SAMPLE = 30;

struct Some {
	int a;
	int b;
	int aPos;
	int bPos;

	Some(int a, int b, int aPos, int bPos){
		a = a;
		b = b;
		aPos = aPos;
		bPos = bPos;
	}
};


__device__ int GetIndex(char c) {
	if (a[i] == 'a')
			return 0;
	if (a[i] == 'c')
		return 1;
	if (a[i] == 't')
		return 2;
	if (a[i] == 'g')
		return 3;
}

__device__ int GetInt (char* a, int len, long long int * value) {
	for (int i = 0; i < len; ++i) {
		value[0] <<= 1;
		value[1] <<= 1;
		value[2] <<= 1;
		value[3] <<= 1;
		value[GetIndex(a[i])] |= 1;
	}
	return 1 << (len - 1);
}

__device__ int BitapProcess (char* a, int aLen, char *b, int bLen) {
	long long int mask[4];

	int aLimit = GetInt(a, len, *mask);

	long long int res[MAX_ERROR];

	bool flag = false;
	for (int i = 0; i < bLen; ++i) {
		long long int lastValue = res[0];
		res[0] <<= 1;
		res[0] &= mask[GetIndex(b[i])];
		for (int t = 1; t <= MAX_ERROR; ++t) {
			long long int insert = lastValue;
			long long int del = (res[t - 1] << 1);
			long long int replace = (lastValue << 1);
			lastValue = res[t];
			res[t] <<= 1;
			res[t] &= mask[GetIndex(b[i])] | insert | del | replace;

			//значение eps 0.125
			if (i + 1 >= MIN_LEN_FOR_SAMPLE && t <  0.25 * (i + 1)) {
				//todo
				//нашли вхождение, нужно как-нибудь запомнить
				return i + 1;
			}
		}
	}
	//todo
	//если не нашли ничего подходящего пока возвращаем 0
	return 0;
}

__global__ void bitap (char * a, char * b, int * aPos, int * bPos, int* data) {
	int bx = threadIdx.x;
	int y = blockIdx.x;
	int x = bx + y * BLOCK_SIZE;
	int shift_ans = 0;
	__shared__ char parts [BLOCK_SIZE][4 * 64];
	
	int aN = aPos[data[4 * x]];
	int aP = data[4 * x + 2];
	int ALen = aPos[data[4 * x] + 1] - aN;

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
		else alenFor = 0;
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

	int left = BitapProcess(parts[bx][0], aLenRev, parts[bx][64 + i], bLenRev);
	int right = BitapProcess(parts[bx][2 * 64], aLenFor, parts[bx][64 * 3 + i], bLenFor);

	if (left != 0 || right != 0) {
		data[4 * x + 2] = bP - left;
		data[4 * x + 3] = bp + K_MER_LEN + right;
	} else {
		data[4 * x + 2] = -1;
		data[4 * x + 3] = -1;
	}
} 

__int64 hash_degree;
void CalcHashDegree() {
	hash_degree = 1;
	for (int i = 0; i < K_MER_LEN; ++i) {
		hash_degree *= HASH_COEFF;
	}
}

char *A;
char *B;
int *_aPos;
int *_bPos;
int _counter;

map<__int64, map<int, vector<int>>> k_mers_in_b;
map<pair<int, int>, int[DIAG_COUNT]> result;
vector<Some> toCuda;

void AddNewItem(int a, int b, int aPos, int bPos) {
	pair<int, int> p = make_pair(a, b);
	if (result.count(p) == 0) {
		result.insert(p, new int[DIAG_COUNT]);
	}
	int d1 = (aPos - bPos) / DIAG_COUNT;
	int d2 = d1 + 1;
	++result[p][d1];
	++result[p][d2];
	if (result[p][d2] == COMMON_PLACE) {
		toCuda.push_back(new Some(a, b, aPos, bPos));
	}
}

int maxLenA;
int maxLenB;
int countA;
int countB;

int main() {
	ReadData();
	ProcessData();	
	int *toGpu = new int[toCuda.size() * 4];
	for (int i = 0; i < toCuda.size(); ++i) {
		toGpu[i * 4] = toCuda[i].a;
		toGpu[i * 4 + 1] = toCuda[i].b;
		toGpu[i * 4 + 2] = toCuda[i].aPos;
		toGpu[i * 4 + 3] = toCuda[i].bPos;
	}

	char * devA = NULL;
	char * devB = NULL;
	int * devAPos = NULL;
	int * devBPos = NULL;
	int * devCheckIt = NULL;

	cudaMalloc ( (void**)&devA, maxLenA * sizeof (char));
	cudaMalloc ( (void**)&devB, maxLenB * sizeof (char));
	cudaMalloc ( (void**)&devAPos, countA * sizeof(int));
	cudaMalloc ( (void**)&devBPos, countB * sizeof (int));
	cudaMalloc ( (void**)&devCheckIt, toCuda.size() * 4 * sizeof (int));

	dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3((int) ((m_host - 0.5) / threads.x) + 1); 

	cudaMemcpy (devA, A,  maxLenA * sizeof (char), cudaMemcpyHostToDevice);
	cudaMemcpy (devB, B,  maxLenB * sizeof (char), cudaMemcpyHostToDevice);
	cudaMemcpy (devAPos, _aPos,  countA * sizeof (int), cudaMemcpyHostToDevice);
	cudaMemcpy (devBPos, _bPos,  countB * sizeof (int), cudaMemcpyHostToDevice);
	cudaMemcpy (devCheckIt, toGpu,  toCuda.size() * 4 * sizeof (int), cudaMemcpyHostToDevice);
	
	bitap <<<blocks, threads>>> (devA, devB, devAPos, devBPos, devCheckIt);

	cudaDeviceSynchronize();

	cudaMemcpy (toGpu, devCheckIt, toCuda.size() * 4 * sizeof (int), cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost);

	for (int i = 0; i < toCuda.size(); ++i) {
		if (toGpu[i * 4 + 2] >= 0) {
			printf ("A:%d B:%d BPosStart:%d BPosFinish:%d", toGpu[i * 4] + 1, toGpu[i * 4 + 1] + 1, toGpu[i * 4 + 2] + 1, toGpu[i * 4 + 3] + 1);
		}
	}
}

void ReadData() {
	freopen("data.txt", "rt" ,stdin);
	freopen("answer.txt", "wt", stdout);
	//считываем блок А
	scanf("%d%d", &countA, &maxLenA);
	//пусть будет общее кол-во символов(максимальное)
	A = new char[maxLenA];
	_aPos = new int[countA];
	_aPos[0] = 0;
	for (int i = 0; i < countA; ++i) {
		int len;
		scanf("%d", &len);
		if (i != countA - 1)
			_aPos[i + 1] = _aPos[i] + len;
		for (int j = 0; j < len; ++j) {
			char t;
			do {
				t = getchar();
			} while (!isalpha(t));
			A[_aPos[i] + j] = t;
		}
	}
	//считываем блок B
	scanf("%d%d", &countB, &maxLenB);
	B = new char[maxLenB];
	_bPos = new int[countB];
	_bPos[0] = 0;
	for (int i = 0; i < countB; ++i) {
		int len;
		scanf("%d", &len);
		if (i != countB - 1)
			_bPos[i + 1] = _bPos[i] + len;
		for (int j = 0; j < len; ++j) {
			char t;
			do {
				t = getchar();
			} while (!isalpha(t));
			B[_bPos[i] + j] = t;
		}
	}
}

void ProcessData() {
	CalcHashDegree();
	for (int i = 0; i < countB; ++i) {
		__int64 currentHash = 0;
		for (int j = 0; j < K_MER_LEN; ++j) {
			currentHash = currentHash * HASH_COEFF + (B[_bPos[i] + j] - 'A' + 1);
		}
		if (!k_mers_in_b.count(currentHash)) {
			k_mers_in_b.insert(make_pair(currentHash, new map<int, vector<int>>));
		}
		if (!k_mers_in_b[currentHash].count(i)) {
			k_mers_in_b[currentHash].insert(make_pair(i, new vector<int>));
		}
		k_mers_in_b[currentHash][i].push_back(0);
		for (int j = 1; j < _bPos[i + 1] - _bPos[i] - K_MER_LEN + 1; ++j) {
			currentHash = (currentHash / hash_degree) * HASH_COEFF + (B[_bPos[i] + j + K_MER_LEN - 1] - 'A' + 1);
			if (!k_mers_in_b.count(currentHash)) {
				k_mers_in_b.insert(make_pair(currentHash, new map<int, vector<int>>));
			}
			if (!k_mers_in_b[currentHash].count(i)) {
				k_mers_in_b[currentHash].insert(make_pair(i, new vector<int>));
			}
			k_mers_in_b[currentHash][i].push_back(j);
		}
		
	}
	for (int i = 0; i < countA; ++i) {
		__int64 currentHash = 0;
		for (int j = 0; j < K_MER_LEN; ++j) {
			currentHash = currentHash * HASH_COEFF + (A[_aPos[i] + j] - 'A' + 1);
		}
		map<int, vector<int>>::iterator iter = k_mers_in_b[currentHash].begin();
		for(;iter != k_mers_in_b[currentHash].end(); ++iter) {
			for (int t = 0; t < iter->second.size(); ++t) {
				AddNewItem(i, iter->first, 0, iter->second[t]);
			}
		}
		for (int j = 1; j < _aPos[i + 1] - _aPos[i] - K_MER_LEN + 1; ++j) {
			currentHash = (currentHash / hash_degree) * HASH_COEFF + (A[_aPos[i] + j + K_MER_LEN - 1] - 'A' + 1);
			map<int, vector<int>>::iterator iter = k_mers_in_b[currentHash].begin();
			for(;iter != k_mers_in_b[currentHash].end(); ++iter) {
				for (int t = 0; t < iter->second.size(); ++t) {
					AddNewItem(i, iter->first, j, iter->second[t]);
				}
			}
		}
	}
}