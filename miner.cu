#include "miner.h"
#include "equihash.h"
#include <pthread.h>
#include <stdio.h>

void minerInitialize() {
	// Initialize the equihash solver.
	equihashInitialize();
}

void minerStart(void (*getBlockHeader)(unsigned int *outBlockHeader), void solutionFound(unsigned int *solution)) {
	// Randomize nonce.
	srand(time(NULL));
	size_t const nonceSizeInWords = 8;
	unsigned int nonce[nonceSizeInWords];
	for (size_t i = 0; i < nonceSizeInWords; i++) {
		nonce[i] = rand();
	}

	// Allocate space for the block header.
	unsigned int *blockHeader = (unsigned int *)malloc(108);

	// Allocate enough space for the solutions.
	unsigned int *solutions = (unsigned int *)malloc(100 * 512 * sizeof(unsigned int));

	unsigned int totalNumberOfSolutions = 0;
	unsigned int numberOfRounds = 0;

	while (1) {
		// Get the most up-to-date block header.
		getBlockHeader(blockHeader);

		// Solve the equihash problem.
		unsigned int numberOfSolutions = equihashSolve(blockHeader, nonce, solutions);
		if (numberOfSolutions > 0) {
			// Post the solutions.
			for (size_t i = 0; i < numberOfSolutions; i++) {
				solutionFound(solutions + i * 512 *sizeof(unsigned int));
			}
		}

		// Generate next nonce.
		nonce[0]++;

		// Statistics.
		totalNumberOfSolutions += numberOfSolutions;
		numberOfRounds++;
		printf("Average solutions per round: %f\n", (double)totalNumberOfSolutions / numberOfRounds);
	}
}
