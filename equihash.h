#ifndef EQUIHASH_H_
#define EQUIHASH_H_

void equihashInitialize();

unsigned int equihashSolve(unsigned int *blockHeader, unsigned int *nonce, unsigned int *outSolutions);

#endif /* EQUIHASH_H_ */
