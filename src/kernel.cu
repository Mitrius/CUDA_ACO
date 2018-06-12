#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "vector.h"

const half ACOalpha = 0.2;//increase
const half ACOdelta = 0.1;//decrease
const half ACOgamma = 0.1;//minval
const int block_size = 10;
const int thread_size = 32;

__device__ half& tmat(half *matrix, size_t x, size_t y){
	return x>y ? matrix[y*N+x] : matrix[x*N+y];
}
__device__ char& tmat(char *matrix, size_t x, size_t y){
	return x>y ? matrix[y*N+x] : matrix[x*N+y];
}

__global__ void clique_kernel(size_t *A, size_t N, char *device_graph, half *device_pheromone, curandState *states, unsigned int seed) {
	size_t id = blockIdx.x*blockDim.x+threadIdx.x;
	actual_device_vector<size_t > B, C;
    curand_init(seed*id, id, 0, &states[id]);
	size_t  startIdx = (1-curand_uniform(&states[id]))*N;
	size_t current = startIdx;
	for(size_t i = 0; i < N; i++) if(i!=startIdx && tmat(device_graph, startIdx, i)) B.push_back(i);
	C.push_back(startIdx); //END SETUP
	while(B.size()>0){ //MAIN LOOP
		half norm = 0;
		for(size_t i = 0; i < B.size(); norm += tmat(device_pheromone, current, B[i++]));
		size_t chosen = 0;
		float radom = (1-curand_uniform(&states[id]));
		while(chosen < B.size() && radom<=0) radom-= tmat(device_pheromone, current, B[chosen++], N)/norm; //NEXT VERTEX PICKED
		for(size_t i = 0; i < B.size(); i++) if(i==chosen || !tmat(device_graph, chosen, B[i])) B.erase(i); //REMOVE NON-NEIGHBORING
		current = chosen;
		C.push_back(chosen);
	}
	for(size_t i = 0; i < C.size()-1; i++) tmat(device_pheromone, C[i], C[i+1])=tmat(device_pheromone, C[i], C[i+1])+(half)ACOalpha;
	A[id]=C.size();
}
__global__ void evaporation_kernel(size_t N, half *device_pheromone){
	size_t id = blockIdx.x *blockDim.x + threadIdx.x;
	if(id<N*N) device_pheromone[id] = device_pheromone[id]<= ACOgamma+ACOdelta ? (half)ACOgamma : device_pheromone[id]-(half)ACOdelta;
}
extern "C" int anthill(char **graph, size_t N, size_t M){
	curandState *states;
	char *device_graph;
	half *device_pheromone;
	size_t *results, *host_results=new size_t[block_size*thread_size], max = 0;
	cudaMalloc(&states, block_size*thread_size*sizeof(curandState));
	
	cudaMalloc(&device_graph, N*N*sizeof(char));
	cudaMalloc(&device_pheromone, N*N*sizeof(unsigned short));
	cudaMemcpy(device_graph, graph[0], N*N*sizeof(char), cudaMemcpyHostToDevice); //graph initialized
	
	cudaMalloc(&results, block_size*thread_size*sizeof(size_t));
	for(size_t i = 0; i < M; i++){
		evaporation_kernel<<<N*N/thread_size,thread_size>>>(N, device_pheromone);
		clique_kernel<<<block_size,thread_size>>>(results, N, device_graph, device_pheromone, states, (unsigned int)time(NULL));
		cudaMemcpy(host_results, results, block_size*thread_size*sizeof(size_t), cudaMemcpyDeviceToHost);
		for(size_t j = 0; j < block_size; j++) if(max<host_results[j]) max = host_results[j];
	}
	delete[] host_results;
	cudaFree(results);
	
	cudaFree(device_pheromone);
	cudaFree(device_graph);
	cudaFree(states);
	return max;
}