#ifndef BLAKE2B_H_
#define BLAKE2B_H_

// @param blockHeader Must be 108 bytes.
// @param nonce Must be 32 bytes.
// @params rows Must be preallocated. Will be filled with the initial rows.
// @returns The number of rows created.
unsigned int blake2bFillInitialRows(unsigned int *blockHeader, unsigned int *nonce, unsigned int *rows);

#endif /* BLAKE2B_H_ */
