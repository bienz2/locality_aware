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
    if (comm->local_comm == MPI_COMM_NULL) 
    {
        MPIL_Comm_topo_init(comm);
    }

    int num_procs;
    int local_rank, ppn;
    int num_nodes;
    MPI_Comm_size(comm->global_comm, &num_procs);
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);
    MPI_Comm_size(comm->group_comm, &num_nodes);

    int send_nnz = topo->outdegree;
    int send_size = 0;
    for (int i = 0; i < send_nnz; i++) 
    {
        send_size += sendcounts[i];
    }
    int recv_nnz = topo->indegree;  

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    // 1. Find inter-node receive sizes per-process
    int local_proc, node, size, idx;
    std::vector<int> metadata;  
    std::vector<int> metadata_ctr(ppn, 0);
    std::vector<int> metadata_displs(ppn + 1);
    std::vector<int> dest_l(ppn);
    std::iota(dest_l.begin(), dest_l.end(), 0);

    for (int i = 0; i < topo->indegree; i++) 
    {
        local_proc = get_local_proc(comm, topo->sources[i]);
        metadata_ctr[local_proc] += 2;  
    }

    metadata_displs[0] = 0;
    for (int i = 0; i < ppn; i++) 
    {
        metadata_displs[i + 1] = metadata_displs[i] + metadata_ctr[i];
        metadata_ctr[i] = 0;
    }

    int total_size = metadata_displs[ppn];  
    if (total_size) metadata.resize(total_size);

    for (int i = 0; i < topo->indegree; i++) 
    {
        node = get_node(comm, topo->sources[i]);
        local_proc = get_local_proc(comm, topo->sources[i]);
        size = recvcounts[i];
        idx = metadata_displs[local_proc] + metadata_ctr[local_proc];
        metadata[idx]     = node;
        metadata[idx + 1] = size;
        metadata_ctr[local_proc] += 2;
    }
    for (int i = 0; i < ppn; i++) 
    {
        metadata_ctr[i] = metadata_displs[i + 1] - metadata_displs[i];
    }

    MPIL_Comm* local_lcomm;
    MPIL_Comm_init(&local_lcomm, comm->local_comm);

    // Dynamic communication to find aggregated recv sizes
    int agg_recv_nnz, agg_recv_size;
    int *src_tmp, *recvcounts_tmp, *rdispls_tmp;
    char* recvvals_tmp;
    alltoallv_crs_personalized_dense(ppn, 
            total_size, 
            dest_l.data(),
            metadata_ctr.data(),
            metadata_displs.data(),
            MPI_INT,
            metadata.data(),
            &agg_recv_nnz, 
            &agg_recv_size,
            &src_tmp, 
            &recvcounts_tmp,
            &rdispls_tmp,
            MPI_INT, 
            (void**)&recvvals_tmp, 
            xinfo, 
            local_lcomm);
    agg_recv_size /= 2;
    for (int i = 0; i < agg_recv_nnz; i++)
    {
        recvcounts_tmp[i] /= 2;
        rdispls_tmp[i + 1] /= 2;
    }

    // 2. Aggregated Recvs: 
    std::vector<int> agg_node_idx(num_nodes, -1);
    int agg_node_recvs = 0;
    std::vector<int> agg_node_list;
    std::vector<int> agg_node_sizes;

    int* node_and_size = (int*)recvvals_tmp;
    for (int i = 0; i < agg_recv_size; i++) 
    {
        node = node_and_size[2 * i];
        size = node_and_size[2 * i + 1];
        if (agg_node_idx[node] == -1) {
            agg_node_idx[node] = agg_node_recvs;
            agg_node_list.push_back(node);
            agg_node_sizes.push_back(0);
            agg_node_recvs++;
        }
        agg_node_sizes[agg_node_idx[node]] += size;
    }
    std::vector<int> agg_node_displs(agg_node_recvs + 1);
    agg_node_displs[0] = 0;
    for (int i = 0; i < agg_node_recvs; i++) 
    {
        agg_node_displs[i + 1] = agg_node_displs[i] + agg_node_sizes[i];
    }

    int total_recv_size = agg_node_displs[agg_node_recvs];
    std::vector<char> agg_recv_buf(total_recv_size * recv_bytes);

    std::vector<MPI_Request> recv_requests(agg_node_recvs);
    for (int i = 0; i < agg_node_recvs; i++)
    {
        node = agg_node_list[i];
        int global_proc = get_global_proc(comm, node, local_rank);
        MPI_Irecv(&agg_recv_buf[agg_node_displs[i] * recv_bytes],
                  agg_node_sizes[i],
                  recvtype, global_proc, 0, comm->global_comm, &recv_requests[i]);
    }


    // 3. Aggregated Sends: one message per destination node
    std::vector<int> node_idx(num_nodes, -1);  // FIX: init to -1 for "unseen" check
    int node_sends = 0;
    std::vector<int> node_list(topo->outdegree);
    std::vector<int> node_sizes(topo->outdegree, 0);

    for (int i = 0; i < topo->outdegree; i++) {
        node = get_node(comm, topo->destinations[i]);
        if (node_idx[node] == -1) {  // FIX: check -1 not node_sizes[node]==0
            node_idx[node] = node_sends;
            node_list[node_sends] = node;  // FIX: was node_list[num_sends]
            node_sends++;
        }
        node_sizes[node_idx[node]] += sendcounts[i];
    }

    std::vector<int> node_displs(node_sends + 1);
    node_displs[0] = 0;
    for (int i = 0; i < node_sends; i++) {
        node_displs[i + 1] = node_displs[i] + node_sizes[i];
        node_sizes[i] = 0;
    }

    std::vector<char> agg_buf(send_size * send_bytes);  // FIX: use send_size computed above

    // Sort destinations by local_proc so packing order matches the ascending-rank
    // read order used in step 5 (alltoallv_crs_personalized_dense returns src in
    // ascending rank order).
    std::vector<int> dest_order(topo->outdegree);
    std::iota(dest_order.begin(), dest_order.end(), 0);
    std::sort(dest_order.begin(), dest_order.end(), [&](int a, int b) {
        return get_local_proc(comm, topo->destinations[a])
             < get_local_proc(comm, topo->destinations[b]);
    });

    for (int ii = 0; ii < topo->outdegree; ii++) {
        int i = dest_order[ii];
        node = get_node(comm, topo->destinations[i]);
        idx = node_idx[node];
        memcpy(&agg_buf[(node_displs[idx] + node_sizes[idx]) * send_bytes],
               &send_buffer[sdispls[i] * send_bytes],
               sendcounts[i] * send_bytes);
        node_sizes[idx] += sendcounts[i];
    }

    // Post all sends
    std::vector<MPI_Request> requests(node_sends);
    for (int i = 0; i < node_sends; i++) {
        node = node_list[i];
        int global_proc = get_global_proc(comm, node, local_rank);
        MPI_Isend(&agg_buf[node_displs[i] * send_bytes],
                  node_displs[i + 1] - node_displs[i],
                  sendtype, global_proc, 0, comm->global_comm, &requests[i]);
    }

    // 4. Wait for inter-messages to complete
    MPI_Waitall(agg_node_recvs, recv_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(node_sends, requests.data(), MPI_STATUSES_IGNORE);


    // 5. Redistribute aggregated recv data on-node

    // Compute redist send counts (in elements) per local peer
    std::vector<int> local_sendcounts(ppn, 0);
    for (int i = 0; i < agg_recv_nnz; i++)
    {
        int local_proc  = src_tmp[i];
        int idx  = rdispls_tmp[i];
        int count = recvcounts_tmp[i];
        int* node_and_size = (int*)recvvals_tmp + idx * 2;
        for (int j = 0; j < count; j++)
            local_sendcounts[local_proc] += node_and_size[2 * j + 1];
    }
    std::vector<int> local_sdispls(ppn + 1, 0);
    for (int i = 0; i < ppn; i++)
    {
        local_sdispls[i + 1] = local_sdispls[i] + local_sendcounts[i];
        local_sendcounts[i] = 0;
    }
    int local_size = local_sdispls[ppn];
    std::vector<char> local_sendbuf(local_size * recv_bytes);

    std::vector<int> node_ctr(agg_node_recvs, 0);
    for (int i = 0; i < agg_recv_nnz; i++)
    {
        int local_proc = src_tmp[i];
        int idx = rdispls_tmp[i];
        int count = recvcounts_tmp[i];
        int* node_and_size = (int*)recvvals_tmp + idx * 2;
        for (int j = 0; j < count; j++)
        {
            node = node_and_size[2*j];
            size = node_and_size[2*j+1];
            idx = agg_node_idx[node];
            memcpy(&local_sendbuf[(local_sdispls[local_proc] + local_sendcounts[local_proc]) * recv_bytes],
                    &agg_recv_buf[(agg_node_displs[idx] + node_ctr[idx]) * recv_bytes],
                    size * recv_bytes);
            local_sendcounts[local_proc] += size;
            node_ctr[idx] += size;
        }
    }
    std::vector<int> local_recvcounts(ppn, 0);
    for (int i = 0; i < topo->indegree; i++)
    {
        local_proc = get_local_proc(comm, topo->sources[i]);
        local_recvcounts[local_proc] += recvcounts[i];
    }
    std::vector<int> local_rdispls(ppn+1);
    local_rdispls[0] = 0;
    for (int i = 0; i < ppn; i++)
        local_rdispls[i+1] = local_rdispls[i] + local_recvcounts[i];
    std::vector<char> local_recvbuf(local_rdispls[ppn] * recv_bytes);

    MPI_Alltoallv(local_sendbuf.data(), local_sendcounts.data(), local_sdispls.data(), recvtype,
            local_recvbuf.data(), local_recvcounts.data(), local_rdispls.data(), recvtype,
            comm->local_comm);

    std::vector<int> counter(ppn, 0);
    for (int i = 0; i < topo->indegree; i++)
    {
        local_proc = get_local_proc(comm, topo->sources[i]);
        memcpy(&recv_buffer[rdispls[i] * recv_bytes],
                &local_recvbuf[(local_rdispls[local_proc] + counter[local_proc]) * recv_bytes],
                recvcounts[i] * recv_bytes);
        counter[local_proc] += recvcounts[i];
    }


    MPIL_Comm_free(&local_lcomm);
    MPIL_Info_free(&xinfo);
    MPIL_Free(src_tmp);
    MPIL_Free(recvcounts_tmp);
    MPIL_Free(rdispls_tmp);
    MPIL_Free(recvvals_tmp);

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif

