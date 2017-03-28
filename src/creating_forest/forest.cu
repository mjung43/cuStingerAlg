/********************************************************
 * Forest
 ********************************************************/

#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"

#include <stdio.h>
#include "forest-tester.h"

using namespace cuStingerAlgs;


bcTree* createTreeHost(int64_t numVertices) {
	bcTree* newTree = (bcTree*) allocHostArray(1, sizeof(bcTree));
	newTree->NV = numVertices;
	newTree->vArr = (bcV*) allocHostArray(numVertices, sizeof(bcV));
	
	return newTree;
}

bcTree* createTreeDevice(int64_t numVertices) {
	bcTree* newTree_h = createTreeHost(numVertices);

	/**
	 * due to problem below, i will be trying something new
	 */

	bcV* vArr_h = newTree_h->vArr;
	bcV* vArr_d = (bcV*) allocDeviceArray(numVertices, sizeof(bcV));
	copyArrayHostToDevice(newTree_h->vArr, vArr_d, numVertices, sizeof(bcV));
	newTree_h->vArr = vArr_d;

	// above is something new

	bcTree* newTree_d = (bcTree*) allocDeviceArray(1, sizeof(bcTree));

	copyArrayHostToDevice(newTree_h, newTree_d, 1, sizeof(bcTree));
	// destroyTreeHost(newTree_h);

	/**
	 * Is the vArr of device struct pointing to vArr of host struct???
	 * because when destroying / freeing host struct, vArr from host is also freed and so
	 * vArr in device tampered with, displaying wrong data
	 */

	// still testing this
	newTree_h->vArr = vArr_h;
	destroyTreeHost(newTree_h);

	return newTree_d;
}

bcTree* copyTreeHostToDevice(bcTree* tree_h) {
	bcV* vArr_h = tree_h->vArr;
	int64_t numVertices = tree_h->NV;

	bcV* vArr_d = (bcV*) allocDeviceArray(numVertices, sizeof(bcV));
	copyArrayHostToDevice(tree_h->vArr, vArr_d, numVertices, sizeof(bcV));
	tree_h->vArr = vArr_d;

	bcTree* newTree_d = (bcTree*) allocDeviceArray(1, sizeof(bcTree));

	copyArrayHostToDevice(tree_h, newTree_d, 1, sizeof(bcTree));

	tree_h->vArr = vArr_h;

	return newTree_d;
}

bcTree* copyTreeDeviceToHost(bcTree* tree_d, int64_t numVertices) {
	bcTree* newTree_h = createTreeHost(numVertices);
	bcV* vArr_h = newTree_h->vArr;
	copyArrayDeviceToHost(tree_d, newTree_h, 1, sizeof(bcTree));

	copyArrayDeviceToHost(newTree_h->vArr, vArr_h, numVertices, sizeof(bcV));

	newTree_h->vArr = vArr_h;

	return newTree_h;
}

void destroyTreeHost(bcTree* tree_h) {
	freeHostArray(tree_h->vArr);
    freeHostArray(tree_h);
}

void destroyTreeDevice(bcTree* tree_d) {
	freeDeviceArray(tree_d);
}

int main(int argc, char** argv) {
	//code
	int64_t numVertices = 10;
	bcTree* tree = createTreeHost(numVertices);
	bcV* vArr = tree->vArr;
	vArr[0] = 5;
	vArr[1] = 8;

	bcTree* newTree_d = copyTreeHostToDevice(tree);

	bcTree* newTree_h = copyTreeDeviceToHost(newTree_d, numVertices);

	printf("%ld\n%ld\n%ld\n", newTree_h->NV, newTree_h->vArr[0], newTree_h->vArr[1]);
}

