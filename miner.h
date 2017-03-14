#ifndef MINER_H_
#define MINER_H_

void minerInitialize();

void minerStart(void (*getBlockHeader)(unsigned int *outBlockHeader), void solutionFound(unsigned int *solution));

#endif /* MINER_H_ */
