#include <iostream>
#include <string>
#include <cstdio>
#include <unordered_map>
#include <cstdlib>
#include <mpi.h>

#define INCOMING_DATA 0
#define END_PROCESS 2
#define GET_DATA 1

const std::string kFilePath = "./data/facebook_clean_data/";
const std::string kFeatures[]{"athletes_edges.csv", "company_edges.csv", "government_edges.csv"
,"politician_edges.csv","public_figure_edges.csv", "tvshow_edges.csv"};
const unsigned short kAmountOfNodes = 14113;
const int iteration_count = 100;

extern "C" int anthill(unsigned short *graph, int N, int M);

void ReceiveAndCalculate()
{

    MPI_Status status;
    MPI_Request request;
    unsigned short result = 0;
    unsigned short *edges = new unsigned short[kAmountOfNodes * kAmountOfNodes];
    int message_size;

    MPI_Isend(&result, 1, MPI_UNSIGNED_SHORT, 0, GET_DATA, MPI_COMM_WORLD, &request);
    for (;;)
    {
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_UNSIGNED_SHORT, &message_size);
        MPI_Recv(edges, message_size, MPI_UNSIGNED_SHORT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == END_PROCESS)
        {
            std::cout << " process received order to commit sudoku\n";
            break;
        }
        result = anthill(edges, kAmountOfNodes, iteration_count);
        MPI_Isend(&result, 1, MPI_UNSIGNED_SHORT, 0, INCOMING_DATA, MPI_COMM_WORLD, &request);
    }
    delete[] edges;
}
void LoadCSVs(int process_count)
{
    std::string row = "";
    std::unordered_map<int, std::string> ongoming_comps;

    char trash;

    int index = 0;
    unsigned short result = -1;

    MPI_Request request;
    MPI_Status status;
    FILE *f_handle;

    unsigned short *edges = new unsigned short[kAmountOfNodes * kAmountOfNodes];

    for (std::string file_name : kFeatures)
    {
        index = 0;
        MPI_Irecv(&result, 1, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);

        f_handle = fopen((kFilePath + file_name).c_str(), "rt");
        while (!feof(f_handle))
        {
            fscanf(f_handle, "%d,%d\n", &edges[index], &edges[index + 1]);
            index += 2;
        }
        fclose(f_handle);

        MPI_Wait(&request, &status);

        if (status.MPI_TAG == INCOMING_DATA)
        {
            std::cout << "file: " << ongoming_comps[status.MPI_SOURCE]
                      << " clique size: " << result << '\n';
            ongoming_comps.erase(status.MPI_SOURCE);
        }

        MPI_Isend(edges, index, MPI_UNSIGNED_SHORT, status.MPI_SOURCE, INCOMING_DATA, MPI_COMM_WORLD, &request);

        std::cout << "Send data from file: " + file_name + '\n';
        ongoming_comps[status.MPI_SOURCE] = file_name;
    }
    while (!ongoming_comps.empty())
    {
        MPI_Irecv(&result, 1, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        std::cout << "file: " << ongoming_comps[status.MPI_SOURCE]
                  << " clique size: " << result << '\n';
        ongoming_comps.erase(status.MPI_SOURCE);
    }
    for (int i = 1; i < process_count; i++)
    {
        MPI_Isend(&result, 1, MPI_UNSIGNED_SHORT, i, END_PROCESS, MPI_COMM_WORLD, &request);
        std::cout << "told process with id: " << i << " to kill himself\n";
    }

    delete[] edges;
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
