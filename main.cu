#include <stdio.h>
#include "miner.h"

void getBlockHeader(unsigned int *outBlockHeader) {
	memcpy(outBlockHeader, "D12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678912345678", 108);
}

void solutionFound(unsigned int *solution) {

}

int main() {
	// Initialize the miner.
	minerInitialize();

	// Start the miner.
	minerStart(getBlockHeader, solutionFound);

	printf("\n");
}
