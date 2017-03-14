#include "equihash.h"
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <openssl/sha.h>
#include "blake2b.h"
#include "cudaUtils.h"

size_t const n = 200;
size_t const k = 9;
size_t const collisionLength = n / (k + 1);
size_t const hashSizeInWords = k + 1;
size_t const hashPaddingInWords = 6; // Such that hashSizeInWords + hashPaddingInWords % 32 == 0
size_t const intermediaryIndexesRowSizeInWords = 4;
size_t const maximumNumberOfRows = 2400000U;
size_t const numberOfBuckets = 1 << collisionLength;
size_t const bucketWidthInWords = 16;
size_t const solutionSizeInWords = 1 << k;
size_t const maximumNumberOfSolutions = 100;

static unsigned int *rows0, *rows1 = NULL;
static unsigned int *counters = NULL;
static unsigned int *buckets = NULL;
static unsigned int *intermediaryIndexes = NULL;
static unsigned int *solutions = NULL;

#define intermediaryIndexesAfterRound(intermediaryIndexes, round) ((intermediaryIndexes) + (round) * maximumNumberOfRows * intermediaryIndexesRowSizeInWords)

#include <time.h>
#include <sys/time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void equihashInitialize() {
	// Create 2 buffers to alternate between.
	size_t const rowsSize = maximumNumberOfRows * (hashSizeInWords + hashPaddingInWords) * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc(&rows0, rowsSize));
	checkCudaErrors(cudaMalloc(&rows1, rowsSize));

	// Create the counters.
	checkCudaErrors(cudaMalloc(&counters, numberOfBuckets * sizeof(unsigned int)));

	// Create the buckets.
	checkCudaErrors(cudaMalloc(&buckets, numberOfBuckets * bucketWidthInWords * sizeof(unsigned int)));

	// Create buffers to store row indexes for every intermediary round.
	checkCudaErrors(cudaMalloc(&intermediaryIndexes, k * maximumNumberOfRows * intermediaryIndexesRowSizeInWords * sizeof(unsigned int)));

	// Create buffer to store the solutions.
	checkCudaErrors(cudaMalloc(&solutions, maximumNumberOfSolutions * solutionSizeInWords * sizeof(unsigned int)));
}

__global__ void countAndDistributeCollisionsToBuckets(unsigned int *rows, unsigned int numberOfRows, unsigned int *counters, unsigned int *buckets, size_t round) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numberOfRows) {
		unsigned int subhashValue = rows[i * (hashSizeInWords + hashPaddingInWords)];
		unsigned int oldCount = atomicAdd(counters + subhashValue, 1);
		if (oldCount >= bucketWidthInWords) {
			printf("bigger\n");
//			printf("hash: %u %u %u collisions = %u indexes = %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n", subhashValue, rows[i * (hashSizeInWords + (1 << round)) + round + 1], rows[i * (hashSizeInWords + (1 << round)) + round + 2], oldCount, rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 1], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 2], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 3], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 4], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 5], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 6], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 7], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 8], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 9], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 10], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 11], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 12], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 13], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 14], rows[i * (hashSizeInWords + (1 << round)) + hashSizeInWords + 15]);
		} else {
			buckets[oldCount * numberOfBuckets + subhashValue] = i;
		}
	}
}

__device__ unsigned int numberOfRows = 0;

__global__ void createNewRows(unsigned int *oldRows, unsigned int *newRows, unsigned int *counters, unsigned int *buckets, unsigned int *intermediaryIndexes, size_t round) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int numberOfCollisions = counters[i];
	// If at least 2 duplicates exist, pair-wise select duplicates to create new rows.
	if (numberOfCollisions >= 2) {
		for (size_t firstBucketValueIndex = 0; firstBucketValueIndex < numberOfCollisions - 1; firstBucketValueIndex++) {
			for (size_t secondBucketValueIndex = firstBucketValueIndex + 1; secondBucketValueIndex < numberOfCollisions; secondBucketValueIndex++) {
				unsigned int firstBucketValue = buckets[firstBucketValueIndex * numberOfBuckets + i];
				unsigned int secondBucketValue = buckets[secondBucketValueIndex * numberOfBuckets + i];
				unsigned int *firstRow = oldRows + firstBucketValue * (hashSizeInWords + hashPaddingInWords);
				unsigned int *secondRow = oldRows + secondBucketValue * (hashSizeInWords + hashPaddingInWords);
				// Check if the child is likely to contain duplicate indexes based on the following heuristic technique.
				unsigned int childRowIsLikelyAllZeros = (firstRow[1] ^ secondRow[1]) == 0; // TODO: Optimize me, reuse result.
				if (!childRowIsLikelyAllZeros) {
					// Check if the parent's first index is the same.
					// Load the first index of both rows.
					unsigned int firstIndexOfFirstRow;
					unsigned int firstIndexOfSecondRow;
					if (round == 0) { // TODO: Optimize me.
						firstIndexOfFirstRow = firstBucketValue;
						firstIndexOfSecondRow = secondBucketValue;
					} else {
						firstIndexOfFirstRow = (intermediaryIndexesAfterRound(intermediaryIndexes, round - 1) + firstBucketValue * intermediaryIndexesRowSizeInWords)[0];
						firstIndexOfSecondRow = (intermediaryIndexesAfterRound(intermediaryIndexes, round - 1) + secondBucketValue * intermediaryIndexesRowSizeInWords)[0];
					}
					// Proceed only if the indexes are not the same.
					if (firstIndexOfFirstRow != firstIndexOfSecondRow) {
						// Copy the xor'ed hashes.
						unsigned int newRowIndex = atomicAdd(&numberOfRows, 1);
						unsigned int *newRow = newRows + newRowIndex * (hashSizeInWords + hashPaddingInWords);
						for (size_t subhashIndex = 1; subhashIndex < hashSizeInWords - round; subhashIndex++) {
							newRow[subhashIndex - 1] = firstRow[subhashIndex] ^ secondRow[subhashIndex]; // TODO: Optimize me, read/write 64bit instead.
						}
						// Swap the indexes if required by the canonical order.
						if (firstIndexOfFirstRow > firstIndexOfSecondRow) {
							// Swap indexes.
							unsigned int temp;
							temp = firstIndexOfFirstRow;
							firstIndexOfFirstRow = firstIndexOfSecondRow;
							firstIndexOfSecondRow = temp;
							// Swap bucket values.
							temp = firstBucketValue;
							firstBucketValue = secondBucketValue;
							secondBucketValue = temp;
						}
						// Write the indexes.
						unsigned int *indexesRow = (intermediaryIndexesAfterRound(intermediaryIndexes, round) + newRowIndex * intermediaryIndexesRowSizeInWords);
						indexesRow[0] = firstIndexOfFirstRow;
						indexesRow[2] = firstBucketValue; // TODO: Optimize me? 64bit write?
						indexesRow[3] = secondBucketValue;
					}
				}
			}
		}
	}
}

__global__ void createLastRoundRows(unsigned int *oldRows, unsigned int *newRows, unsigned int *counters, unsigned int *buckets, unsigned int *intermediaryIndexes) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int numberOfCollisions = counters[i];
	// If at least 2 duplicates exist, pair-wise select duplicates to create new rows.
	if (numberOfCollisions >= 2) {
		for (size_t firstBucketValueIndex = 0; firstBucketValueIndex < numberOfCollisions - 1; firstBucketValueIndex++) {
			for (size_t secondBucketValueIndex = firstBucketValueIndex + 1; secondBucketValueIndex < numberOfCollisions; secondBucketValueIndex++) {
				unsigned int firstBucketValue = buckets[firstBucketValueIndex * numberOfBuckets + i];
				unsigned int secondBucketValue = buckets[secondBucketValueIndex * numberOfBuckets + i];
				unsigned int *firstRow = oldRows + firstBucketValue * (hashSizeInWords + hashPaddingInWords);
				unsigned int *secondRow = oldRows + secondBucketValue * (hashSizeInWords + hashPaddingInWords);
				// Check if the last `collisionLength` bits of the hash are the same.
				unsigned int potentialSolution = (firstRow[1] ^ secondRow[1]) == 0;
				if (potentialSolution) {
					// Check if the parent's first index is the same.
					// Load the first index of both rows.
					unsigned int firstIndexOfFirstRow = (intermediaryIndexesAfterRound(intermediaryIndexes, k - 2) + firstBucketValue * intermediaryIndexesRowSizeInWords)[0];
					unsigned int firstIndexOfSecondRow = (intermediaryIndexesAfterRound(intermediaryIndexes, k - 2) + secondBucketValue * intermediaryIndexesRowSizeInWords)[0];
					// Proceed only if the indexes are not the same.
					if (firstIndexOfFirstRow != firstIndexOfSecondRow) {
						// Copy the xor'ed hashes.
						unsigned int newRowIndex = atomicAdd(&numberOfRows, 1);
						// Swap the indexes if required by the canonical order.
						if (firstIndexOfFirstRow > firstIndexOfSecondRow) {
							// Swap bucket values.
							unsigned int temp;
							temp = firstBucketValue;
							firstBucketValue = secondBucketValue;
							secondBucketValue = temp;
						}
						// Write the indexes.
						unsigned int *indexesRow = (intermediaryIndexesAfterRound(intermediaryIndexes, k - 1) + newRowIndex * intermediaryIndexesRowSizeInWords);
						indexesRow[2] = firstBucketValue; // TODO: optimize me? 64bit write?
						indexesRow[3] = secondBucketValue;
					}
				}
			}
		}
	}
}

__device__ void recursiveReconstructIndexes(unsigned int *intermediaryIndexes, unsigned int round, unsigned int rowIndex, unsigned int *solutions) {
	unsigned int *intermediaryIndexesRow = intermediaryIndexesAfterRound(intermediaryIndexes, round) + rowIndex * intermediaryIndexesRowSizeInWords;
	unsigned int firstIntermediaryIndex = intermediaryIndexesRow[2];
	unsigned int secondIntermediaryIndex = intermediaryIndexesRow[3];

	if (round == 0) {
		solutions[0] = firstIntermediaryIndex;
		solutions[1] = secondIntermediaryIndex;
	} else {
		recursiveReconstructIndexes(intermediaryIndexes, round - 1, firstIntermediaryIndex, solutions);
		recursiveReconstructIndexes(intermediaryIndexes, round - 1, secondIntermediaryIndex, solutions + (1 << round));
	}
}

// Could be optimized?
__global__ void reconstructIndexes(unsigned int *intermediaryIndexes, unsigned int *solutions) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	recursiveReconstructIndexes(intermediaryIndexes, k - 1, i, solutions + i * solutionSizeInWords);
}

unsigned int equihashSolve(unsigned int *blockHeader, unsigned int *nonce, unsigned int *outSolutions) {
	// Setup timer.
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    cudaEvent_t start, stop;
    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

	// Start the timer.
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

	// Create the initial table.
    unsigned int hostNumberOfRows = blake2bFillInitialRows(blockHeader, nonce, rows0);

    char currentRowBufferIndex = 0;

    for (size_t round = 0; round < k - 1 /* the last round needs to be handled differently... */; round++) {
//    	printf("Round #%zu\n", round + 1);
//    	printf("Number of rows before %d\n", hostNumberOfRows);

    	// Clean counters.
    	checkCudaErrors(cudaMemset(counters, 0, numberOfBuckets * sizeof(unsigned int)));

    	// Get the current rows buffer.
    	unsigned int *oldRows = currentRowBufferIndex ? rows1 : rows0;
    	unsigned int *newRows = currentRowBufferIndex ? rows0 : rows1;

    	// Count and distribute collisions to buckets.
    	unsigned int numberOfThreadsPerBlock = 128;
    	unsigned int numberOfBlocks = ceil((double)hostNumberOfRows / numberOfThreadsPerBlock);
    	countAndDistributeCollisionsToBuckets<<<numberOfBlocks, numberOfThreadsPerBlock>>>(oldRows, hostNumberOfRows, counters, buckets, round);

    	// Reset the number of rows.
    	unsigned int zero = 0;
    	checkCudaErrors(cudaMemcpyToSymbol(numberOfRows, &zero, sizeof(typeof(zero)), 0, cudaMemcpyHostToDevice));

    	// Create the new rows.
    	numberOfThreadsPerBlock = 32; // For some reasons, 32 is better than 128 here.
    	numberOfBlocks = numberOfBuckets / numberOfThreadsPerBlock;
    	createNewRows<<<numberOfBlocks, numberOfThreadsPerBlock>>>(oldRows, newRows, counters, buckets, intermediaryIndexes, round);

    	checkCudaErrors(cudaMemcpyFromSymbol(&hostNumberOfRows, numberOfRows, sizeof(typeof(numberOfRows)), 0, cudaMemcpyDeviceToHost));

//    	printf("Number of rows after  %d\n", hostNumberOfRows);

    	// Change the current rows buffer index.
    	currentRowBufferIndex = !currentRowBufferIndex;
    }

    // Last round.
    {
//    	printf("Round #%zu\n", k);
//    	printf("Number of rows before %d\n", hostNumberOfRows);

    	// Clean counters.
    	checkCudaErrors(cudaMemset(counters, 0, numberOfBuckets * sizeof(unsigned int)));

    	// Get the current rows buffer.
    	unsigned int *oldRows = currentRowBufferIndex ? rows1 : rows0;

    	// Count and distribute collisions to buckets.
    	unsigned int numberOfThreadsPerBlock = 128;
    	unsigned int numberOfBlocks = ceil((double)hostNumberOfRows / numberOfThreadsPerBlock);
    	countAndDistributeCollisionsToBuckets<<<numberOfBlocks, numberOfThreadsPerBlock>>>(oldRows, hostNumberOfRows, counters, buckets, k - 1);

    	// Reset the number of rows.
    	unsigned int zero = 0;
    	checkCudaErrors(cudaMemcpyToSymbol(numberOfRows, &zero, sizeof(typeof(zero)), 0, cudaMemcpyHostToDevice));

    	// Create the new rows.
    	unsigned int *newRows = currentRowBufferIndex ? rows0 : rows1;
    	numberOfThreadsPerBlock = 32; // For some reasons, 32 is better than 128 here.
    	numberOfBlocks = numberOfBuckets / numberOfThreadsPerBlock;
    	createLastRoundRows<<<numberOfBlocks, numberOfThreadsPerBlock>>>(oldRows, newRows, counters, buckets, intermediaryIndexes);

    	checkCudaErrors(cudaMemcpyFromSymbol(&hostNumberOfRows, numberOfRows, sizeof(typeof(numberOfRows)), 0, cudaMemcpyDeviceToHost));

//    	printf("Number of rows after  %d\n", hostNumberOfRows);
    }

    unsigned int numberOfValidSolutions = 0;
    if (hostNumberOfRows > 0) {
        // Reconstruct indexes.
    	reconstructIndexes<<<1, hostNumberOfRows>>>(intermediaryIndexes, solutions);

    	// Copy all the solutions.
        checkCudaErrors(cudaMemcpy(outSolutions, solutions, hostNumberOfRows * (1 << k) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Final duplicates check on the CPU for now.
//        double t1 = get_wall_time();
        for (size_t s = 0; s < hostNumberOfRows; s++) {
        	unsigned int *solution = outSolutions + s * 512;
        	unsigned char validSolution = 1;
        	for (size_t i1 = 0; (i1 < 512 - 1) && validSolution; i1++) {
        		for (size_t i2 = i1 + 1; (i2 < 512) && validSolution; i2++) {
        			if (solution[i1] == solution[i2]) {
        				validSolution = 0;
        			}
        		}
        	}
        	if (validSolution) {
        		// Copy the valid solution
        		memcpy(outSolutions + numberOfValidSolutions * 512, solution, 512 * sizeof(unsigned int));
            	numberOfValidSolutions++;
        	}
        }
//        double t2 = get_wall_time();
//        printf("duplicates check: %f\n", t2 - t1);
    }

	// Stop the timer.
    checkCudaErrors(cudaEventRecord(stop, 0));
    sdkStopTimer(&timer);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    sdkDeleteTimer(&timer);
    printf("Elapsed time: %f\n", elapsedTimeInMs/1000);

    return numberOfValidSolutions;
;
}
