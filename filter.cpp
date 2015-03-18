#define _CRT_SECURE_NO_WARNINGS
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include "bitap.cuh"

using namespace std;

const int HASH_COEFF = 5;


const int DIAG_COUNT = 32;
const int COMMON_PLACE = 1;

struct Some {
	int a;
	int b;
	int aPos;
	int bPos;

	Some(int aT, int bT, int aPosT, int bPosT){
		a = aT;
		b = bT;
		aPos = aPosT;
		bPos = bPosT;
	}
};

__int64 hash_degree;
void CalcHashDegree() {
	hash_degree = 1;
	for (int i = 0; i < K_MER_LEN - 1; ++i) {
		hash_degree *= HASH_COEFF;
	}
}

char *A;
char *B;
int *_aPos;
int *_bPos;
int _counter;

map<__int64, map<int, vector<int>>> k_mers_in_b;
map<pair<int, int>, int*> result;
vector<Some> toCuda;

void AddNewItem(int a, int b, int aPos, int bPos) {
	pair<int, int> p = make_pair(a, b);
	if (result.count(p) == 0) {
		result.insert(make_pair(p, new int[DIAG_COUNT]));
		for (int i = 0; i < DIAG_COUNT; ++i) {
			result[p][i] = 0;
		}
	}
	int d1 = (aPos - bPos) / DIAG_COUNT;
	int d2 = d1 + 1;
	++result[p][d1];
	++result[p][d2];
	if (result[p][d2] == COMMON_PLACE || result[p][d1] == COMMON_PLACE) {
		Some term(a, b, aPos, bPos);
		toCuda.push_back(term);
	}
}

int maxLenA;
int maxLenB;
int countA;
int countB;

char GetIndex(char c) {
	if (c == 'a')
		return 0;
	if (c == 'c')
		return 1;
	if (c == 't')
		return 2;
	if (c == 'g')
		return 3;
	return 0;
}

void ReadData() {
	freopen("data.txt", "rt" ,stdin);
	freopen("answer.txt", "wt", stdout);
	//считываем блок ј
	scanf("%d%d", &countA, &maxLenA);
	//пусть будет общее кол-во символов(максимальное)
	A = new char[maxLenA];
	_aPos = new int[countA + 1];
	_aPos[0] = 0;
	for (int i = 0; i < countA; ++i) {
		int len;
		scanf("%d", &len);
		_aPos[i + 1] = _aPos[i] + len;
		for (int j = 0; j < len; ++j) {
			char t;
			do {
				t = getchar();
			} while (!isalpha(t));
			A[_aPos[i] + j] = GetIndex(t);
		}
	}
	//считываем блок B
	scanf("%d%d", &countB, &maxLenB);
	B = new char[maxLenB + 1];
	_bPos = new int[countB];
	_bPos[0] = 0;
	for (int i = 0; i < countB; ++i) {
		int len;
		scanf("%d", &len);
		_bPos[i + 1] = _bPos[i] + len;
		for (int j = 0; j < len; ++j) {
			char t;
			do {
				t = getchar();
			} while (!isalpha(t));
			B[_bPos[i] + j] = GetIndex(t);
		}
	}
}

void ProcessData() {
	CalcHashDegree();
	for (int i = 0; i < countB; ++i) {
		__int64 currentHash = 0;
		for (int j = 0; j < K_MER_LEN; ++j) {
			currentHash = currentHash * HASH_COEFF + B[_bPos[i] + j] + 1;
		}
		if (!k_mers_in_b.count(currentHash)) {
			auto m = new map<int, vector<int>>();
			pair <__int64, map <int, vector<int>>> p(currentHash, *m);
			k_mers_in_b.insert(p);
		}
		if (!k_mers_in_b[currentHash].count(i)) {
			auto v = new vector<int>();
			k_mers_in_b[currentHash].insert(make_pair(i, *v));
		}
		k_mers_in_b[currentHash][i].push_back(0);
		for (int j = 1; j < _bPos[i + 1] - _bPos[i] - K_MER_LEN + 1; ++j) {
			currentHash = (currentHash % hash_degree) * HASH_COEFF + B[_bPos[i] + j + K_MER_LEN - 1] + 1;
			if (!k_mers_in_b.count(currentHash)) {
				auto m = new map<int, vector<int>>();
				k_mers_in_b.insert(make_pair(currentHash, *m));
			}
			if (!k_mers_in_b[currentHash].count(i)) {
				auto v = new vector<int>();
				k_mers_in_b[currentHash].insert(make_pair(i, *v));
			}
			k_mers_in_b[currentHash][i].push_back(j);
		}
		
	}
	for (int i = 0; i < countA; ++i) {
		__int64 currentHash = 0;
		for (int j = 0; j < K_MER_LEN; ++j) {
			currentHash = currentHash * HASH_COEFF + (A[_aPos[i] + j] + 1);
		}
		map<int, vector<int>>::iterator iter = k_mers_in_b[currentHash].begin();
		for(;iter != k_mers_in_b[currentHash].end(); ++iter) {
			for (unsigned int t = 0; t < iter->second.size(); ++t) {
				AddNewItem(i, iter->first, 0, iter->second[t]);
			}
		}
		for (int j = 1; j < _aPos[i + 1] - _aPos[i] - K_MER_LEN + 1; ++j) {
			currentHash = (currentHash % hash_degree) * HASH_COEFF + (A[_aPos[i] + j + K_MER_LEN - 1] + 1);
			map<int, vector<int>>::iterator iter = k_mers_in_b[currentHash].begin();
			for(;iter != k_mers_in_b[currentHash].end(); ++iter) {
				for (unsigned int t = 0; t < iter->second.size(); ++t) {
					AddNewItem(i, iter->first, j, iter->second[t]);
				}
			}
		}
	}
}

int main() {
	ReadData();
	ProcessData();	
	int *toGpu = new int[toCuda.size() * 4];
	for (unsigned int i = 0; i < toCuda.size(); ++i) {
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
	int * devAnswer = NULL;

	cudaMalloc ( (void**)&devA, maxLenA * sizeof (char));
	cudaMalloc ( (void**)&devB, maxLenB * sizeof (char));
	cudaMalloc ( (void**)&devAPos, (countA + 1) * sizeof(int));
	cudaMalloc ( (void**)&devBPos, (countB + 1) * sizeof (int));
	cudaMalloc ( (void**)&devCheckIt, toCuda.size() * 4 * sizeof (int));
	cudaMalloc ( (void**)&devAnswer, toCuda.size() * 6 * sizeof (int));

	cudaMemcpy (devA, A,  maxLenA * sizeof (char), cudaMemcpyHostToDevice);
	cudaMemcpy (devB, B,  maxLenB * sizeof (char), cudaMemcpyHostToDevice);
	cudaMemcpy (devAPos, _aPos,  (countA + 1) * sizeof (int), cudaMemcpyHostToDevice);
	cudaMemcpy (devBPos, _bPos,  (countB + 1) * sizeof (int), cudaMemcpyHostToDevice);
	cudaMemcpy (devCheckIt, toGpu,  toCuda.size() * 4 * sizeof (int), cudaMemcpyHostToDevice);

	dim3 threads = dim3(BLOCK_SIZE);
	dim3 blocks  = dim3((int) ((toCuda.size() - 0.5) / threads.x) + 1); 

	Bitap(blocks,threads, devA, devB, devAPos, devBPos, devCheckIt, devAnswer, toCuda.size());

	cudaDeviceSynchronize();

	auto toGpu2 = new int[toCuda.size() * 6];
	cudaMemcpy(toGpu2, devAnswer, toCuda.size() * 6 * sizeof (int), cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < toCuda.size(); ++i) {
		if (toGpu2[i * 6] >= 0) {
			printf ("A:%d B:%d APosStart:%d APosFinish%d BPosStart:%d BPosFinish:%d\n",
				toGpu2[i * 6], toGpu2[i * 6 + 1], toGpu2[i * 6 + 2], toGpu2[i * 6 + 3],toGpu2[i * 6 + 4], toGpu2[i * 6 + 5]);
		}
	}
}
