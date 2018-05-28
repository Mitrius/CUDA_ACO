#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <unordered_map>
#include <cstdlib>
#include <ctime> 
#include <cuda.h>
#include <curand.h>
#include <mpich-x86_64/mpi.h>

#define INCOMING_DATA 0
#define END_PROCESS 2
#define GET_DATA 1

const std::string kFilePath = "./data/facebook_clean_data/";
const std::string kFeatures[]{"athletes_edges.csv", "company_edges.csv", "government_edges.csv"};
const unsigned int kAmountOfNodes = 14113;

const float alpha = 0.2;//increase
const float delta = 0.1;//decrease
const float gamma = 0.1;//minval
const int block_size = 10;
const int iteration_count = 10000;


__global__ void clique_kernel(int startIdx, int *A, int N, char **device_graph, float **device_pheromone, curandState *state) {
	thrust::device_vector<int> C(N/2), B(N/2);
	for(int i = 0; i < N; i++) if(graph[startIdx][i]) B.push_back(i);
	C.push_back(startIdx); //END SETUP
	int current = startIdx;
	while(B.size()>0){ //MAIN LOOP
		float norm = 0.0f;
		for(int i = 0; i < B.size(); i++) norm += device_pheromone[current][B[i]];
		int chosen = 0;
		float radom = curand_uniform(&state);
		for(chosen=0;chosen < B.size() || radom<=0;chosen++,radom-= device_pheromone[current][B[chosen]]/norm); //NEXT VERTEX PICKED
		for(int i = 0; i < B.size(); i++) if(!graph[chosen][B[i]]) B.erase(B.begin()+i); //REMOVE NON-NEIGHBORING
		current = chosen;
		C.push_back(chosen);
	}
	for(int i = 0; i < C.size()-1; i++) {
		device_pheromone[C[i]][C[i+1]]=device_pheromone[C[i]][C[i+1]]+alpha;
		device_pheromone[C[i+1]][C[i]]=device_pheromone[C[i]][C[i+1]];
	}
	A[blockIdx.x*blockDim.x]=C.size();
}
__global__ void evaporation_kernel(int N, float **device_pheromone){
	int row = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i = 0; i<N; i++) {
		device_pheromone[row][i] -= delta;
		if(device_pheromone[row][i]<gamma) device_pheromone[row][i]=gamma;
	}
}
__host__ int anthill(char **graph, int N, int M){
	curandState *state;
	char **device_graph;
	float **device_pheromone;
	srand(time(NULL));
	cudaMalloc(&state, sizeof(curandState));
	cudaMalloc(&device_graph, N*sizeof(char*));
	cudaMalloc(&device_pheromone, N*sizeof(float*));
    curand_init(time(NULL), i, 0, state);
	void **temp = malloc(N*sizeof(char*)), **temp2 = malloc(N*sizeof(float*));
	for(int i = 0; i < N; i++) {
		cudaMalloc(&temp[i], N*sizeof(char));
		cudaMemcpy(temp[i], graph[i], N, cudaMemcpyHostToDevice);
		cudaMalloc(&temp2[i], N*sizeof(float));
		cudaMemset(temp2[i], gamma, N);
	}
	cudaMemcpy(device_graph, temp, N, cudaMemcpyHostToDevice); //graph initialized
	cudaMemcpy(device_pheromone, temp2, N, cudaMemcpyHostToDevice); //device_pheromone initialized
	int *results, *host_results=malloc(block_size*sizeof(int)), max = 0;
	cudaMalloc(&results, block_size*sizeof(int));
	for(int i = 0; i < M; i++){
		clique_kernel<<<block_size,1>>>(rand()%N, results, N, device_graph, device_pheromone, state);
		evaporation_kernel<<<N,1>>>(N, device_pheromone);
		cudaMemcpy(host_results, results, N, cudaMemcpyDeviceToHost);
		for(int i = 0; i < block_size; i++) if(max<host_results[i]) max = host_results[i];
	}
	cudaFree(results);
	for(int i = 0; i < N; i++){
		cudaFree(temp[i]);
		cudaFree(temp2[i]);
	}
	cudaFree(device_pheromone);
	cudaFree(device_graph);
	cudaFree(state);
	return max;
}
char **createArray()
{
    char **arr = new char *[kAmountOfNodes];
    int size = kAmountOfNodes * kAmountOfNodes;
    arr[0] = new char[size];
    for (int j = 1; j < kAmountOfNodes; j++)
    {
        arr[j] = &arr[0][j * kAmountOfNodes];
    }
    return arr;
}
void ReceiveAndCalculate()
{
    char **map = createArray();
    MPI_Status status;
    MPI_Request request;
    int result = -1;
    MPI_Isend(&result, 1, MPI_INT, 0, GET_DATA, MPI_COMM_WORLD, &request);
    for(;;)
    {
        MPI_Recv(&(map[0][0]), kAmountOfNodes * kAmountOfNodes, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == END_PROCESS)
            break;
		result = anthill(map, kAmountOfNodes, iteration_count);
        MPI_Isend(&result, 1, MPI_INT, 0, INCOMING_DATA, MPI_COMM_WORLD, &request);
    }
    delete[] map;
}
void LoadCSVs(int process_count)
{
    std::string row = "";
    std::unordered_map<int, std::string> ongoming_comps;

    char trash;
    char **map = createArray();

    int a, b;
    int result = -1;

    MPI_Request request;
    MPI_Status status;
    std::cout << (sizeof(map[0][0]) * (kAmountOfNodes * kAmountOfNodes)) / (1024 * 1024) << '\n';
    for (std::string file_name : kFeatures)
    {
        std::ifstream file_in(kFilePath + file_name);
        if (file_in.is_open())
        {
            MPI_Irecv(&result, 1, MPI_INTEGER, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
            while (file_in >> a >> trash >> b && trash == ',')
                map[a][b] = map[b][a] = 1;
            MPI_Wait(&request, &status);
            if (status.MPI_TAG == INCOMING_DATA)
            {
                std::cout << "file: " << ongoming_comps[status.MPI_SOURCE]
                          << " clique size: " << result << '\n';
            }
            MPI_Isend(&(map[0][0]), kAmountOfNodes * kAmountOfNodes, MPI_CHAR, status.MPI_SOURCE, INCOMING_DATA, MPI_COMM_WORLD, &request);
            std::cout << "Send data from file: " + file_name + '\n';
            ongoming_comps[status.MPI_SOURCE] = file_name;
            file_in.close();
            std::fill(&map[0][0], &map[0][0] + sizeof(map), 0);
        }
    }
    for (int i = 0; i < process_count; i++)
        MPI_Isend(&(map[0][0]), kAmountOfNodes * kAmountOfNodes, MPI_CHAR, i, END_PROCESS, MPI_COMM_WORLD, &request);
    delete[] map;
}
int main(int argc, char **argv)
{
    int process_rank, process_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    if (!process_rank)
    {
        LoadCSVs(process_count);
    }
    else
    {
        ReceiveAndCalculate();
    }

    MPI_Finalize();
    return 0;
}
