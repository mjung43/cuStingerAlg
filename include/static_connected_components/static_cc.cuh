#pragma once

#include "algs.cuh"

// Connected Components

namespace cuStingerAlgs {

class ccData {

public:
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
};

class StaticConnectedComponents:public StaticAlgorithm {
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

	ccData hostCCData, *deviceCCData;

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

	static __device__ __forceinline__ void setLevelAndCount(cuStinger* custing, vertexId_t src, void* metadata) {
		ccData* data = (ccData*) metadata;
		data->level[src] = INT32_MAX;
		data->count[src] = 0;
	}

	static __device__ __forceinline__ void findNextVertex(cuStinger* custing, vertexId_t src, void* metadata) {
		ccData* data = (ccData*) metadata;
		if (data->level[src] == INT32_MAX) {
			if (data->level[data->root] != INT32_MAX || src < data->root) {
				data->root = src;
			}
		}
	}

};

}