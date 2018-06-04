#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <unordered_map>
#include <cstdlib>
#include <mpi.h>

#define INCOMING_DATA 0
#define END_PROCESS 2
#define GET_DATA 1

const std::string kFilePath = "./data/facebook_clean_data/";
const std::string kFeatures[]{"athletes_edges.csv", "company_edges.csv", "government_edges.csv"};
const unsigned int kAmountOfNodes = 14113;
const int iteration_count = 10000;

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
