#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "macros.cuh"

#include "streaming_connected_components/cc.cuh"


using namespace std;

namespace cuStingerAlgs {


void StreamingConnectedComponents::Init(cuStinger& custing) {

	hostCCData.num = 0;

	hostCCData.nv = custing.nv;

	hostCCData.thresh = 4;

	hostCCData.queue.Init(custing.nv);
	hostCCData.next.Init(custing.nv);
	hostCCData.temp.Init(custing.nv);

	hostCCData.level = (vertexId_t*) allocDeviceArray(hostCCData.nv, sizeof(vertexId_t));

	hostCCData.CId = (vertexId_t*) allocDeviceArray(hostCCData.nv, sizeof(vertexId_t));

	hostCCData.PN = (vertexId_t*) allocDeviceArray(hostCCData.nv * hostCCData.nv, sizeof(vertexId_t));

	hostCCData.count = (length_t*) allocDeviceArray(hostCCData.nv, sizeof(int64_t));

	deviceCCData = (ccData*) allocDeviceArray(1, sizeof(ccData));

	copyArrayHostToDevice(&hostCCData, deviceCCData, 1, sizeof(ccData));

	Reset();
}

void StreamingConnectedComponents::Reset() {
	hostCCData.queue.resetQueue();
	hostCCData.currLevel = INT32_MAX;

	copyArrayHostToDevice(&hostCCData, deviceCCData, 1, sizeof(ccData));
}

void StreamingConnectedComponents::setInputParameters(vertexId_t root) {
	hostCCData.root = root;
}

void StreamingConnectedComponents::Release() {
	freeDeviceArray(deviceCCData);
	freeDeviceArray(hostCCData.level);
}

void StreamingConnectedComponents::Run(cuStinger& custing) {

	// cusLoadBalance cusLB(hostCCData.nv);

	allVinG_TraverseVertices<ccOperator::setArrays>(custing, deviceCCData);

	for (length_t i = 0; i < hostCCData.nv; i++) {
		hostCCData.root = i;
		SyncDeviceWithHost();

		copyArrayDeviceToHost(hostCCData.level + hostCCData.root, &hostCCData.currLevel, 1, sizeof(vertexId_t));

		if (hostCCData.currLevel == INT32_MAX) {
			hostCCData.num++;
			RunBfsTraversal(custing);
		}
		
		Reset();
	}

	printf("Total number of connected components: %d\n", hostCCData.num);

}

void StreamingConnectedComponents::RunBfsTraversal(cuStinger& custing) {

	cusLoadBalance cusLB(hostCCData.nv);
	
	hostCCData.queue.enqueueFromHost(hostCCData.root);

	SyncDeviceWithHost();
	copyArrayHostToDevice(&hostCCData.currLevel, hostCCData.level + hostCCData.root, 1, sizeof(length_t));

	length_t prevEnd = 1;
	while((hostCCData.queue.getActiveQueueSize()) > 0){

		allVinA_TraverseEdges_LB<ccOperator::ccExpandFrontier>(custing, deviceCCData, cusLB, hostCCData.queue);

		SyncHostWithDevice();
		hostCCData.queue.setQueueCurr(prevEnd);
		prevEnd = hostCCData.queue.getQueueEnd();

		hostCCData.currLevel++;
		SyncDeviceWithHost();
	}
}

void StreamingConnectedComponents::InsertEdges(cuStinger& custing, vertexId_t* s, vertexId_t* d, length_t len) {
	//call insert method in ccoperator
	BatchUpdateData bud(len, true, custing.nv);
	vertexId_t *srcs = bud.getSrc();
	vertexId_t *dsts = bud.getDst();

	// need to put in operator later
	for (length_t i = 0; i < len; i++) {
		srcs[i] = s[i];
		dsts[i] = d[i];
	}

	// inserting edges into custinger
	length_t requireAllocation;
	BatchUpdate bu = BatchUpdate(bud);

	custing.edgeInsertions(bu, requireAllocation);

	// need to do for all new edges <s,t> in E later

	allEinAs_TraverseEdges<ccOperator::insertEdge>(custing, deviceCCData, srcs, dsts, len);

}



}