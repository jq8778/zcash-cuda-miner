#include "blake2b.h"
#include "cudaUtils.h"

const size_t blockHeaderSize = 108;
const size_t nonceSize = 32;
const size_t indexSize = 4;

__device__ __constant__ static const unsigned long long blake2bIV[8] = {
    0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
};

__device__ __constant__ const unsigned char sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

__device__ unsigned long long circularShiftRight64(unsigned long long x, size_t shift) {
    return (x >> shift) ^ (x << (64 - shift));
}


__device__ void G(unsigned long long *v, size_t a, size_t b, size_t c, size_t d, unsigned long long x, unsigned long long y) {
    v[a] = (v[a] + v[b] + x);
    v[d] = circularShiftRight64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = circularShiftRight64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = circularShiftRight64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = circularShiftRight64(v[b] ^ v[c], 63);
}

__device__ void F(unsigned long long *h, unsigned long long *m, unsigned long long t, char flag) {
    // Initialize local work vector v[0..15]
    unsigned long long v[16];
    for (size_t i = 0; i < 8; i++) {
        v[i] = h[i];
        v[8 + i] = blake2bIV[i];
    }

    v[12] = v[12] ^ t;
    if (flag) {
        v[14] = ~v[14];
    }

    // Cryptographic mixing
    for (size_t i = 0; i < 12; i++) {
        // Message word selection permutation for this round.
        G(v, 0, 4,  8, 12, m[sigma[i][0]], m[sigma[i][1]]);
        G(v, 1, 5,  9, 13, m[sigma[i][2]], m[sigma[i][3]]);
        G(v, 2, 6, 10, 14, m[sigma[i][4]], m[sigma[i][5]]);
        G(v, 3, 7, 11, 15, m[sigma[i][6]], m[sigma[i][7]]);
        G(v, 0, 5, 10, 15, m[sigma[i][8]], m[sigma[i][9]]);
        G(v, 1, 6, 11, 12, m[sigma[i][10]], m[sigma[i][11]]);
        G(v, 2, 7,  8, 13, m[sigma[i][12]], m[sigma[i][13]]);
        G(v, 3, 4,  9, 14, m[sigma[i][14]], m[sigma[i][15]]);
    }

    for (size_t i = 0; i < 8; i++) {
        h[i] = h[i] ^ v[i] ^ v[i + 8];
    }
}

// Assumptions:
// - data == 108 (block header) + 32 (nonce) + 4 (index) bytes = 144 (padded up to 256 bytes with zeros).
__device__ void optimizedBlake2b(unsigned long long *data, size_t dataLength, void *firstHalfHashOut, void *secondHalfHashOut, size_t hashLength) {
    // Initialization vector.
    unsigned long long h[8];
    for (size_t i = 0; i < 8; i++) {
        h[i] = blake2bIV[i];
    }

    // Parameter block p[0]
    h[0] = h[0] ^ 0x01010000 ^ hashLength;
    h[6] = h[6] ^ 0x576F50687361635AULL; // "ZcashPoW" personalization.
    h[7] = h[7] ^ 0x9000000C8ULL; // htole(N = 200) | htole(K = 9) personalization.

    // First 128-byte block.
    F(h, data, 128, 0);

    // Second & final block.
    F(h, data + (128 / 8) /* second 128-byte block */, dataLength, 1);

    memcpy(firstHalfHashOut, h, hashLength / 2);
    memcpy(secondHalfHashOut, ((char *)h) + hashLength / 2, hashLength / 2);
}

__device__ __constant__ char *blockHeaderPlusNonce[256];

__global__ void generateRows(unsigned int *rows) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t hashLength = 2 * 200 / 8;
    const size_t dataLength = blockHeaderSize + nonceSize + indexSize;

    char inputData[256];
    memcpy(inputData, blockHeaderPlusNonce, 256);
    *((unsigned int *)(inputData + blockHeaderSize + nonceSize)) = i;

    unsigned int *hashOut1 = rows + 2 * i * 16;
    unsigned int *hashOut2 = rows + 2 * i * 16 + 16;

    optimizedBlake2b((unsigned long long *)inputData, dataLength, hashOut1, hashOut2, hashLength);
}

__global__ void expandRows(unsigned int * const __restrict__ rows) {
	const unsigned int rowIndex = blockDim.y * blockIdx.x + threadIdx.y;
	const unsigned int columnIndex = threadIdx.x;
	const unsigned int rowSizeInSharedMemoryInWords = 8;
	const unsigned int rowSizeInGlobalMemoryInWords = 16;

	// Allocate shared memory to contain the compressed rows.
	__shared__ unsigned int shared[rowSizeInSharedMemoryInWords * 128];

	shared[threadIdx.y * rowSizeInSharedMemoryInWords + columnIndex] = rows[rowIndex * rowSizeInGlobalMemoryInWords + columnIndex];

	__syncthreads();

	if (columnIndex < 5) {
		unsigned char *chunk = (unsigned char *)shared + threadIdx.y * rowSizeInSharedMemoryInWords * sizeof(unsigned int) + 5 * columnIndex;
		unsigned long long expandedChunk = chunk[0] | (chunk[1] << 8) | ((chunk[2] & 0xF0) << 12) | (((unsigned long long)chunk[3]) << 32) | ((unsigned long long)chunk[4] << 40) | ((unsigned long long)(chunk[2] & 0x0F) << 48);
		*(unsigned long long *)(rows + rowIndex * rowSizeInGlobalMemoryInWords + 2 * columnIndex) = expandedChunk;
//		if (rowIndex == 0) {
//			printf("%d %x %x %x %x %x\n", columnIndex, chunk[0], chunk[1], chunk[2], chunk[3], chunk[4]);
//			printf("%d %x %x %x %x %x %x %x %x\n", columnIndex, ((unsigned char *)&expandedChunk)[0], ((unsigned char *)&expandedChunk)[1], ((unsigned char *)&expandedChunk)[2], ((unsigned char *)&expandedChunk)[3], ((unsigned char *)&expandedChunk)[4], ((unsigned char *)&expandedChunk)[5], ((unsigned char *)&expandedChunk)[6], ((unsigned char *)&expandedChunk)[7]);
//		}
	}
}

unsigned int blake2bFillInitialRows(unsigned int *blockHeader, unsigned int *nonce, unsigned int *rows) {
	char *hostBlockHeaderPlusNonce = (char *)calloc(256, 1);
	memcpy(hostBlockHeaderPlusNonce, blockHeader, blockHeaderSize);
	memcpy(hostBlockHeaderPlusNonce + blockHeaderSize, nonce, nonceSize);

    checkCudaErrors(cudaMemcpyToSymbol(blockHeaderPlusNonce, hostBlockHeaderPlusNonce, 256, 0, cudaMemcpyHostToDevice));

	unsigned int numberOfThreadsPerBlock = 128;
	unsigned int numberOfBlocks = (1 << 20) / numberOfThreadsPerBlock;
    generateRows<<<numberOfBlocks, numberOfThreadsPerBlock>>>(rows);

	dim3 numberOfThreadsPerBlock2D(8, 128);
	numberOfBlocks = (1 << 21) / numberOfThreadsPerBlock2D.y;
    expandRows<<<numberOfBlocks, numberOfThreadsPerBlock2D>>>(rows);

    return 1 << 21;
}
