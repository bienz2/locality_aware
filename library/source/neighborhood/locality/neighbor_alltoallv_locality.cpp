#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/alltoall_crs.h"

#ifdef __cplusplus
extern "C" {
#endif

int neighbor_alltoallv_locality(const void* sendbuf, 
        const int sendcounts[], 
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcounts[], 
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIL_Topo* topo, 
        MPIL_Comm* comm)
{
    // If local communicator doesn't exist yet, create it
    if (comm->local_comm == MPI_COMM_NULL) 
    {
        MPIL_Comm_topo_init(comm);
    }

    // Get rank/size for MPI communicators
    int num_procs, rank;
    int local_rank, ppn;
    int num_nodes;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);
    MPI_Comm_size(comm->group_comm, &num_nodes);

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;
    
    int send_size = 0;
    for (int i = 0; i < topo->outdegree; i++) 
        send_size += sendcounts[i];
    int recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
        recv_size += recvcounts[i];

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    /*********************************************************
     ***** 1. Find inter-node receive sizes per-process  *****
     *********************************************************/
    int global_proc, local_proc, node;
    int size, idx, agg_idx;
    int ctr, next_ctr, source_node;
    std::vector<std::pair<int,int>> local_rank_recv_pairs;
    std::vector<int> local_rank_recvcounts(ppn, 0);
    std::vector<int> local_rank_rdispls(ppn+1);
    std::vector<int> local_rank_src(ppn);
    std::iota(local_rank_src.begin(), local_rank_src.end(), 0);

    for (int i = 0; i < topo->indegree; i++)
    {
        local_proc = get_local_proc(comm, topo->sources[i]);
        local_rank_recvcounts[local_proc]++;
    }
    local_rank_rdispls[0] = 0;
    for (int i = 0; i < ppn; i++)
    {
        local_rank_rdispls[i+1] = local_rank_rdispls[i] + local_rank_recvcounts[i];
        local_rank_recvcounts[i] = 0; // reuse this when filling node_and_size array
    }
    int local_rank_recv_size = local_rank_rdispls[ppn];
    if (local_rank_recv_size)
        local_rank_recv_pairs.resize(local_rank_recv_size);

    for (int i = 0; i < topo->indegree; i++) 
    {
        node = get_node(comm, topo->sources[i]);
        local_proc = get_local_proc(comm, topo->sources[i]);
        idx = local_rank_rdispls[local_proc] + local_rank_recvcounts[local_proc];
        local_rank_recv_pairs[idx] = {node, recvcounts[i]};
        local_rank_recvcounts[local_proc]++;
    }

    MPIL_Comm* local_lcomm;
    MPIL_Comm_init(&local_lcomm, comm->local_comm);

    // Dynamic communication to find aggregated recv sizes
    int local_rank_send_num, local_rank_send_size;
    int *local_rank_dest, *local_rank_sendcounts, *local_rank_sdispls;
    std::pair<int,int>* local_rank_send_pairs;
    alltoallv_crs_personalized_dense(ppn, 
            local_rank_recv_size, 
            local_rank_src.data(),
            local_rank_recvcounts.data(),
            local_rank_rdispls.data(),
            MPI_2INT,
            local_rank_recv_pairs.data(),
            &local_rank_send_num, 
            &local_rank_send_size,
            &local_rank_dest, 
            &local_rank_sendcounts,
            &local_rank_sdispls,
            MPI_2INT, 
            (void**)&local_rank_send_pairs, 
            xinfo, 
            local_lcomm);

    /*********************************************************
     ***** 2. Aggregated Inter-Node Receives             *****
     *********************************************************/
    std::vector<int> recv_node_idx(num_nodes, -1);
    int agg_num_recvs = 0;
    std::vector<int> agg_src;
    std::vector<int> agg_recvcounts;

    for (int i = 0; i < local_rank_send_size; i++)
    {
        node = local_rank_send_pairs[i].first;
        if (recv_node_idx[node] == -1)
        {
            recv_node_idx[node] = agg_num_recvs++;
            agg_src.push_back(node);
            agg_recvcounts.push_back(0);
        }
        agg_recvcounts[recv_node_idx[node]] += local_rank_send_pairs[i].second;
    }

    std::vector<int> agg_rdispls(agg_num_recvs+1);
    agg_rdispls[0] = 0;
    for (int i = 0; i < agg_num_recvs; i++)
        agg_rdispls[i+1] = agg_rdispls[i] + agg_recvcounts[i];

    int agg_recv_size = agg_rdispls[agg_num_recvs];
    std::vector<char> agg_recvbuf;
    if (agg_recv_size)
        agg_recvbuf.resize(agg_recv_size * recv_bytes);

    std::vector<MPI_Request> recv_requests;
    if (agg_num_recvs)
        recv_requests.resize(agg_num_recvs);

    for (int i = 0; i < agg_num_recvs; i++)
    {
        node = agg_src[i];
        global_proc = get_global_proc(comm, node, local_rank);
        MPI_Irecv(&agg_recvbuf[agg_rdispls[i] * recv_bytes],
                agg_recvcounts[i], recvtype,
                global_proc, tag, comm->global_comm, &recv_requests[i]);
    }


    /*********************************************************
     ***** 2. Aggregated Inter-Node Sends                *****
     *********************************************************/
    std::vector<int> send_node_idx(num_nodes, -1);
    int agg_num_sends = 0;
    std::vector<int> agg_dest;
    std::vector<int> agg_sendcounts;

    for (int i = 0; i < topo->outdegree; i++)
    {
        node = get_node(comm, topo->destinations[i]);
        if (send_node_idx[node] == -1)
        {
            send_node_idx[node] = agg_num_sends++;
            agg_dest.push_back(node);
            agg_sendcounts.push_back(0);
        }
        agg_sendcounts[send_node_idx[node]] += sendcounts[i];
    }

    std::vector<int> agg_sdispls(agg_num_sends+1);
    agg_sdispls[0] = 0;
    for (int i = 0; i < agg_num_sends; i++)
    {
        agg_sdispls[i+1] = agg_sdispls[i] + agg_sendcounts[i];
        agg_sendcounts[i] = 0; // reset to use during packing
    }

    std::vector<char> agg_sendbuf;
    if (send_size)
        agg_sendbuf.resize(send_size * send_bytes);

    // Sort sends by local rank -- aggregating by node, 
    // recvs assume ordered within by local rank of destination
    std::vector<int> order(topo->outdegree);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), 
            [&](int a, int b)
            {
                return get_local_proc(comm, topo->destinations[a])
                    < get_local_proc(comm, topo->destinations[b]);
            });

    // Pack aggregated messages
    for (int i = 0; i < topo->outdegree; i++)
    {
        idx = order[i];
        node = get_node(comm, topo->destinations[idx]);
        agg_idx = send_node_idx[node];
        memcpy(&agg_sendbuf[(agg_sdispls[agg_idx] + agg_sendcounts[agg_idx]) * send_bytes],
                &send_buffer[sdispls[idx] * send_bytes],
                sendcounts[idx] * send_bytes);
        agg_sendcounts[agg_idx] += sendcounts[idx];
    }

    // Send aggregated messages
    std::vector<MPI_Request> send_requests(agg_num_sends);
    for (int i = 0; i < agg_num_sends; i++)
    {
        node = agg_dest[i];
        global_proc = get_global_proc(comm, node, local_rank);
        MPI_Isend(&agg_sendbuf[agg_sdispls[i] * send_bytes],
                agg_sendcounts[i],
                sendtype,
                global_proc, 
                tag,
                comm->global_comm,
                &send_requests[i]);
    }

    /*********************************************************
     ***** 4. Wait for Inter-Node Messages To Complete   *****
     *********************************************************/
    MPI_Waitall(agg_num_recvs, recv_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(agg_num_sends, send_requests.data(), MPI_STATUSES_IGNORE);


    /*********************************************************
     ***** 5. Redistribute Inter-Node Recvs Locally      *****
     *********************************************************/
    int total_recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
        total_recv_size += recvcounts[i];
    int total_send_size = 0;
    for (int i = 0; i < local_rank_send_size; i++)
        total_send_size += local_rank_send_pairs[i].second;

    std::vector<char> local_sendbuf(total_send_size*recv_bytes);
    std::vector<char> local_recvbuf(total_recv_size*recv_bytes);

    std::vector<int> final_rdispls(ppn+1);
    final_rdispls[0] = 0;
    recv_requests.resize(ppn);
    ctr = 0;
    for (int i = 0; i < ppn; i++)
    {
        size = 0;
        for (int j = local_rank_rdispls[i]; j < local_rank_rdispls[i+1]; j++)
        {
            size += local_rank_recv_pairs[j].second;
        }
        MPI_Irecv(&local_recvbuf[ctr*recv_bytes],
                size,
                recvtype,
                i,
                tag,
                comm->local_comm,
                &recv_requests[i]);
        ctr += size;
        final_rdispls[i+1] = ctr;
    }


    // Which local ranks do I send to
    if (local_rank_send_num)
        send_requests.resize(local_rank_send_num);
    ctr = 0;
    next_ctr = 0;
    for (int i = 0; i < local_rank_send_num; i++)
    {
        local_proc = local_rank_dest[i];

        for (int j = local_rank_sdispls[i]; j < local_rank_sdispls[i+1]; j++)
        {
            source_node = local_rank_send_pairs[j].first;
            agg_idx = recv_node_idx[source_node];
            size = local_rank_send_pairs[j].second;
            memcpy(&local_sendbuf[next_ctr*recv_bytes],
                    &agg_recvbuf[agg_rdispls[agg_idx]*recv_bytes],
                    size*recv_bytes);
            agg_rdispls[agg_idx] += size;
            next_ctr += size;
        }
        MPI_Isend(&local_sendbuf[ctr*recv_bytes],
                next_ctr - ctr,
                recvtype,
                local_proc,
                tag,
                comm->local_comm,
                &send_requests[i]);
        ctr = next_ctr;
    }

    MPI_Waitall(ppn, recv_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(local_rank_send_num, send_requests.data(), MPI_STATUSES_IGNORE);

    /*********************************************************
     ***** 6. Unpack Final Local Receive                 *****
     *********************************************************/
    std::vector<int> local_proc_idx(ppn, -1);
    for (int i = 0; i < ppn; i++)
    {
        local_proc = local_rank_src[i];
        local_proc_idx[local_proc] = i;
    }
    ctr = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        global_proc = topo->sources[i];
        local_proc = get_local_proc(comm, global_proc);
        node = get_node(comm, global_proc);
        idx = local_proc_idx[local_proc];
        memcpy(&recv_buffer[ctr * recv_bytes],
                &local_recvbuf[final_rdispls[idx] * recv_bytes],
                recvcounts[i] * recv_bytes);
        final_rdispls[idx] += recvcounts[i];
        ctr += recvcounts[i];
    }

    MPIL_Comm_free(&local_lcomm);
    MPIL_Info_free(&xinfo);

    MPIL_Free(local_rank_dest);
    MPIL_Free(local_rank_sendcounts);
    MPIL_Free(local_rank_sdispls);
    MPIL_Free(local_rank_send_pairs);

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif

