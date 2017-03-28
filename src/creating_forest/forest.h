/********************************************************
 * forest.h
 ********************************************************/

// forest code from streamingbc

typedef double bc_t;

typedef struct {
    int64_t level;
    int64_t pathsToRoot;
    int64_t edgesBelow;
    int64_t edgesAbove;
    bc_t delta;
} bcV;

typedef struct {
    int64_t NV;
    bcV* vArr;
} bcTree;

typedef bcTree* bcTreePtr;

typedef struct {
    bcTreePtr* forest;
    bc_t* totalBC;
    int64_t NV;
} bcForest;

typedef bcForest* bcForestPtr;



bcTree* createTreeHost(int64_t numVertices);
bcTree* createTreeDevice(int64_t numVertices);
void destroyTreeHost(bcTree* tree_h);
void destroyTreeDevice(bcTree* tree_d);
bcTree* copyTreeHostToDevice(bcTree* tree_h);
