#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "vector.h"

const char ACOalpha = 2;//increase
const char ACOdelta = 1;//decrease
const char ACOgamma = 1;//minval
const int block_size = 10;
const int thread_size = 10;

__global__ void clique_kernel(int *A, int N, char *device_graph, char *device_pheromone, curandState *states, unsigned int seed) {
	int id = blockIdx.x*blockDim.x;
    curand_init(seed*id, id, 0, &states[id]);
	int startIdx = (1-curand_uniform(&states[id]))*N;
	actual_device_vector<int> C(N/2), B(N/2);
	for(int i = 0; i < N; i++) if(i!=startIdx && device_graph[startIdx*N+i]) B.push_back(i);
	C.push_back(startIdx); //END SETUP
	int current = startIdx;
	while(B.size()>0){ //MAIN LOOP
		float norm = 0.0f;
		for(int i = 0; i < B.size(); i++) norm += device_pheromone[current*N+B[i]];
		int chosen = 0;
		float radom = (1-curand_uniform(&states[id]));
		for(chosen=0;chosen < B.size() && radom<=0;chosen++,radom-= device_pheromone[current*N+B[chosen]]/norm); //NEXT VERTEX PICKED
		for(int i = 0; i < B.size(); i++) if(i==chosen || !device_graph[chosen*N+B[i]]) B.erase(i); //REMOVE NON-NEIGHBORING
		current = chosen;
		C.push_back(chosen);
	}
	for(int i = 0; i < C.size()-1; i++) {
		device_pheromone[C[i]*N+C[i+1]]=device_pheromone[C[i]*N+C[i+1]]+ACOalpha;
		device_pheromone[C[i+1]*N+C[i]]=device_pheromone[C[i]*N+C[i+1]];
	}
	A[id]=C.size();
}
__global__ void evaporation_kernel(int N, char *device_pheromone){
	int row = blockIdx.x *blockDim.x + threadIdx.x*thread_size;
	for(int i = 0; i<N/thread_size; i++) {
		device_pheromone[row*N+i] -= ACOdelta;
		if(device_pheromone[row*N+i]<ACOgamma) device_pheromone[row*N+i]=ACOgamma;
	}
}
extern "C" int anthill(char **graph, int N, int M){
	curandState *states;
	char *device_graph;
	char *device_pheromone;
	cudaMalloc(&states, N*sizeof(curandState));
	cudaMalloc(&device_graph, N*N*sizeof(char));
	cudaMalloc(&device_pheromone, N*N*sizeof(char));
	cudaMemcpy(device_graph, graph[0], N*N*sizeof(char), cudaMemcpyHostToDevice); //graph initialized
	int *results, *host_results=new int[block_size*thread_size], max = 0;
	cudaMalloc(&results, block_size*thread_size*sizeof(int));
	for(int i = 0; i < M; i++){
		evaporation_kernel<<<N,thread_size>>>(N, device_pheromone);
		clique_kernel<<<block_size,thread_size>>>(results, N, device_graph, device_pheromone, states, (unsigned int)time(NULL));
		cudaMemcpy(host_results, results, block_size*thread_size*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0; i < block_size; i++) if(max<host_results[i]) max = host_results[i];
	}
	delete[] host_results;
	cudaFree(results);
	cudaFree(device_pheromone);
	cudaFree(device_graph);
	cudaFree(states);
	return max;
}