#pragma once

#include "algs.cuh"
#include "operators.cuh"

// Connected Components

namespace cuStingerAlgs {

class ccData {

public:
	vertexQueue next;
	vertexQueue temp;
	vertexQueue queue;
	vertexId_t* level;
	vertexId_t* CId;
	length_t* count;
	vertexId_t* PN;

	vertexId_t currLevel;
	vertexId_t root;
	length_t nv;
	length_t thresh;

	length_t num;
	bool done;
};

class StreamingConnectedComponents:public StaticAlgorithm {
public:	

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void RunBfsTraversal(cuStinger& custing);


	void SyncHostWithDevice(){
		copyArrayDeviceToHost(deviceCCData, &hostCCData, 1, sizeof(ccData));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostCCData, deviceCCData, 1, sizeof(ccData));
	}

	vertexId_t getLevels() {return hostCCData.currLevel;}
	vertexId_t getElementsFound() {return hostCCData.queue.getQueueEnd();}

	void setInputParameters(vertexId_t root);

	// User is responsible for de-allocating memory.
	vertexId_t* getLevelArrayHost() {
		vertexId_t* hostArr = (vertexId_t*) allocHostArray(hostCCData.nv, sizeof(vertexId_t));
		copyArrayDeviceToHost(hostCCData.level, hostArr, hostCCData.nv, sizeof(vertexId_t) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getLevelArrayForHost(vertexId_t* hostArr) {
		copyArrayDeviceToHost(hostCCData.level, hostArr, hostCCData.nv, sizeof(vertexId_t) );
	}

	void InsertEdges(cuStinger& custing, BatchUpdate& bu, length_t len);

protected:
	ccData hostCCData, *deviceCCData;
private:
    cusLoadBalance* cusLB;

};


class ccOperator:public StaticAlgorithm {
public:

	static __device__ __forceinline__ void ccExpandFrontier(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata) {
		
		ccData* data = (ccData*) metadata;
		vertexId_t nextLevel = data->currLevel + 1;

		vertexId_t prev = atomicCAS(data->level + dst, INT32_MAX, nextLevel); // glorious ternary operator for cuda and updates level
		if (prev == INT32_MAX) {
			data->queue.enqueue(dst);
			data->CId[dst] = data->root;
		}
		if (data->count[dst] < data->thresh) {
			if (data->currLevel < nextLevel) {
				data->PN[data->nv * dst + data->count[dst]] = src;
				data->count[dst] += 1;
			} else if (data->level[src] == data->level[dst]) {
				data->PN[data->nv * dst + data->count[dst]] = -1 * src;
				data->count[dst] += 1; 
			}
		}

	}

	static __device__ __forceinline__ void setArrays(cuStinger* custing, vertexId_t src, void* metadata) {
		ccData* data = (ccData*) metadata;
		data->level[src] = INT32_MAX;
		data->count[src] = 0;
		data->next.enqueue(src);
	}

	static __device__ __forceinline__ void findNextVertex(cuStinger* custing, vertexId_t src, void* metadata) {
		ccData* data = (ccData*) metadata;
		if (data->level[src] == INT32_MAX) {
			data->temp.enqueue(src);
		}
		// change it so next pointer is pointing to temp and temp is pointing to empty next at the end
	}

	static __device__ __forceinline__ void insertEdge(cuStinger* custing, vertexId_t s, vertexId_t d, void* metadata) {
		ccData* data = (ccData*) metadata;
		if (data->CId[s] == data->CId[d]) {
			if (data->level[s] > 0) {
				if (data->level[d] < 0) {
					// d is not safe
					if (data->level[s] < -1 * data->level[d]) {
						if (data->count[d] < data->thresh) {
							data->PN[data->nv * d + data->count[d]] = s;
							data->count[d]++;
						} else {
							data->PN[data->nv * d] = s; 
						}
						data->level[d] *= -1;
					}
				}
			} else {
				if (data->count[d] < data->thresh) {
					if (data->level[s] < data->level[d]) {
						data->PN[data->nv * d + data->count[d]] = s;
						data->count[d]++;
					} else if (data->level[s] == data->level[d]) {
						data->PN[data->nv * d + data->count[d]] = -1 * s;
						data->count[d]++;
					}
				} else if (data->level[s] < data->level[d]) {
					for (length_t i = 0; i < data->thresh; i++) {
						if (data->PN[data->nv * d + i] < 0) {
							data->PN[data->nv * d + i] = s;
							break;
						}
					}
				}
			}

		}
	}

};

template <cusSubKernelEdge cusSK>
static __global__ void device_allEinA_TraverseEdges(cuStinger* custing,void* metadata, BatchUpdateData* bud, int32_t edgesPerThreadBlock){
	length_t e_init=blockIdx.x*edgesPerThreadBlock+threadIdx.x;
	length_t batchSize = *(bud->getBatchSize());
	vertexId_t* d_updatesSrc    = bud->getSrc();
	vertexId_t* d_updatesDst    = bud->getDst();

	for (length_t e_hat=0; e_hat<edgesPerThreadBlock; e_hat+=blockDim.x){
		length_t e=e_init+e_hat;
		if(e>=batchSize){
			break;
		}
		(cusSK)(custing,d_updatesSrc[e],d_updatesDst[e],metadata);
	}
}


template <cusSubKernelEdge cusSK>
static void allEinA_TraverseEdges(cuStinger& custing, void* metadata, BatchUpdate &bu){
	length_t batchSize = *(bu.getHostBUD()->getBatchSize());
	if(batchSize==0)
			return;

	dim3 numBlocks(1, 1); int32_t threads=32;
	dim3 threadsPerBlock(threads, 1);
	int32_t edgesPerThreadBlock=512;

	numBlocks.x = ceil((float)batchSize/(float)edgesPerThreadBlock);
	device_allEinA_TraverseEdges<cusSK><<<numBlocks, threadsPerBlock>>>(custing.devicePtr(), metadata, bu.getDeviceBUD()->devicePtr(), edgesPerThreadBlock);


}




}
