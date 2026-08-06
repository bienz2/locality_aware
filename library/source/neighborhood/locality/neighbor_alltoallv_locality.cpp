#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

// Non-persistent, locality-aware neighbor alltoallv.
//
// Implements the two-step (tap_comm-style) locality-aware communication.
// This is a neighbor collective: every process already knows exactly which
// processes it talks to and how much data is exchanged with each of them
// (topo->destinations/sendcounts and topo->sources/recvcounts), so the
// entire communication schedule below is computed deterministically from
// that information alone -- there is no dynamic/probe-based discovery of
// unknown senders or message sizes anywhere in this routine.
//
//  1. (local, no communication) Each process groups its own outgoing data
//     by destination node.
//  2. (local_comm) A small, fixed-size MPI_Alltoall/MPI_Alltoallv exchanges
//     metadata (which node a piece of data originates from, and how large
//     it is) so that every process learns the exact schedule of data it
//     will relay for its node-mates. This only ever moves small integer
//     bookkeeping, never user data, and every count involved is already
//     known before the calls are made (no MPI_Probe/Iprobe, no Allreduce
//     over all processes).
//  3. (group_comm -- "Global" step) Each process sends, in a single
//     message, all data destined from itself (local rank p on this node)
//     to any process on a remote node m, directly to local rank p on node
//     m. Sizes and partners are exactly known from step 2, so this is
//     plain matched MPI_Isend/MPI_Irecv.
//  4. (local_comm -- "Local_R" step) Each relay process disaggregates the
//     data it just received and forwards it, again via plain matched
//     MPI_Isend/MPI_Irecv (sizes/partners known from steps 1-2), to the
//     processes on its own node that actually requested it.
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

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);
    int num_nodes = comm->num_nodes;
    int my_node   = comm->rank_node;

    // Internal transport is done as raw bytes; sendtype and recvtype are
    // assumed to describe elements of the same width (as is already
    // assumed throughout the rest of the locality-aware routines).
    int item_bytes;
    MPI_Type_size(sendtype, &item_bytes);

    const char* send_buffer = (const char*)sendbuf;
    char* recv_buffer       = (char*)recvbuf;

    /**************************************************************
     * 1. Recipient-side schedule: fully known locally already.
     *    Group my own expected inputs by which local process on my
     *    node will physically receive them off the network (the
     *    "relay" = the process sharing the source's local rank).
     **************************************************************/
    std::vector<std::vector<std::pair<int, int>>> recv_from_relay(ppn);  // {source_node, orig_idx}
    for (int i = 0; i < topo->indegree; i++)
    {
        if (recvcounts[i] == 0)
        {
            continue;
        }
        int source  = topo->sources[i];
        int relay_p = get_local_proc(comm, source);
        int node    = get_node(comm, source);
        recv_from_relay[relay_p].push_back({node, i});
    }
    for (int p = 0; p < ppn; p++)
    {
        std::sort(recv_from_relay[p].begin(), recv_from_relay[p].end());
    }

    std::vector<int> recv_from_relay_size(ppn, 0);
    for (int p = 0; p < ppn; p++)
    {
        for (auto& entry : recv_from_relay[p])
        {
            recv_from_relay_size[p] += recvcounts[entry.second];
        }
    }

    /**************************************************************
     * 2. Local metadata exchange: tell each relay exactly what it
     *    will need to forward on my behalf (source node + count).
     *    Sizes are exchanged first (fixed-size Alltoall), so the
     *    Alltoallv that follows uses only already-known counts.
     **************************************************************/
    std::vector<int> meta_send_counts(ppn, 0);
    for (int p = 0; p < ppn; p++)
    {
        meta_send_counts[p] = (int)recv_from_relay[p].size() * 2;  // (node, count) pairs
    }

    std::vector<int> meta_recv_counts(ppn);
    MPI_Alltoall(meta_send_counts.data(),
                 1,
                 MPI_INT,
                 meta_recv_counts.data(),
                 1,
                 MPI_INT,
                 comm->local_comm);

    std::vector<int> meta_send_displs(ppn + 1, 0);
    std::vector<int> meta_recv_displs(ppn + 1, 0);
    for (int p = 0; p < ppn; p++)
    {
        meta_send_displs[p + 1] = meta_send_displs[p] + meta_send_counts[p];
        meta_recv_displs[p + 1] = meta_recv_displs[p] + meta_recv_counts[p];
    }

    std::vector<int> meta_send_buf(meta_send_displs[ppn]);
    for (int p = 0; p < ppn; p++)
    {
        int off = meta_send_displs[p];
        for (auto& entry : recv_from_relay[p])
        {
            meta_send_buf[off++] = entry.first;               // source node
            meta_send_buf[off++] = recvcounts[entry.second];  // count
        }
    }

    std::vector<int> meta_recv_buf(meta_recv_displs[ppn]);
    MPI_Alltoallv(meta_send_buf.data(),
                  meta_send_counts.data(),
                  meta_send_displs.data(),
                  MPI_INT,
                  meta_recv_buf.data(),
                  meta_recv_counts.data(),
                  meta_recv_displs.data(),
                  MPI_INT,
                  comm->local_comm);

    // Routes I (as a relay) am now responsible for forwarding.
    struct Route
    {
        int source_node;
        int local_q;
        int count;
        int global_offset;  // offset within the Global-step recv buffer for source_node
        int local_offset;   // offset within the Local_R send buffer for local_q
    };
    std::vector<Route> routes;
    for (int q = 0; q < ppn; q++)
    {
        for (int off = meta_recv_displs[q]; off < meta_recv_displs[q + 1]; off += 2)
        {
            Route r;
            r.source_node    = meta_recv_buf[off];
            r.count          = meta_recv_buf[off + 1];
            r.local_q        = q;
            r.global_offset  = 0;
            r.local_offset   = 0;
            routes.push_back(r);
        }
    }

    // Order routes by (source_node, final global destination) -- this
    // matches exactly how the true sender on source_node will pack its
    // outgoing per-node buffer (see step 3), since the final destination's
    // global rank is a value both sides can compute independently.
    std::vector<int> order_by_node(routes.size());
    std::iota(order_by_node.begin(), order_by_node.end(), 0);
    std::sort(order_by_node.begin(), order_by_node.end(), [&](int a, int b) {
        const Route& ra = routes[a];
        const Route& rb = routes[b];
        if (ra.source_node != rb.source_node)
        {
            return ra.source_node < rb.source_node;
        }
        return get_global_proc(comm, my_node, ra.local_q) < get_global_proc(comm, my_node, rb.local_q);
    });

    std::vector<int> recv_node_size(num_nodes, 0);
    for (int idx : order_by_node)
    {
        Route& r        = routes[idx];
        r.global_offset = recv_node_size[r.source_node];
        recv_node_size[r.source_node] += r.count;
    }

    // Order routes by (local_q, source_node) -- this is the layout of the
    // aggregated Local_R message I will send to each local process q; q
    // independently derives the same order from its own recv_from_relay
    // list (both sort by source_node ascending).
    std::vector<int> order_by_q(routes.size());
    std::iota(order_by_q.begin(), order_by_q.end(), 0);
    std::sort(order_by_q.begin(), order_by_q.end(), [&](int a, int b) {
        const Route& ra = routes[a];
        const Route& rb = routes[b];
        if (ra.local_q != rb.local_q)
        {
            return ra.local_q < rb.local_q;
        }
        return ra.source_node < rb.source_node;
    });

    std::vector<int> local_send_size(ppn, 0);
    for (int idx : order_by_q)
    {
        Route& r       = routes[idx];
        r.local_offset = local_send_size[r.local_q];
        local_send_size[r.local_q] += r.count;
    }

    /**************************************************************
     * 3. Global step: gather my own outgoing data by target node,
     *    and exchange it directly with local rank p on that node
     *    (comm->group_comm), with sizes/partners known exactly.
     **************************************************************/
    struct SendChunk
    {
        int dest;
        int count;
        int displ;
    };
    std::vector<std::vector<SendChunk>> send_by_node(num_nodes);
    for (int i = 0; i < topo->outdegree; i++)
    {
        if (sendcounts[i] == 0)
        {
            continue;
        }
        int dest = topo->destinations[i];
        int node = get_node(comm, dest);
        send_by_node[node].push_back({dest, sendcounts[i], sdispls[i]});
    }

    std::vector<int> send_node_size(num_nodes, 0);
    for (int n = 0; n < num_nodes; n++)
    {
        std::sort(send_by_node[n].begin(), send_by_node[n].end(), [](const SendChunk& a, const SendChunk& b) {
            return a.dest < b.dest;
        });
        for (auto& c : send_by_node[n])
        {
            send_node_size[n] += c.count;
        }
    }

    std::vector<int> send_node_displs(num_nodes + 1, 0);
    std::vector<int> recv_node_displs(num_nodes + 1, 0);
    for (int n = 0; n < num_nodes; n++)
    {
        send_node_displs[n + 1] = send_node_displs[n] + send_node_size[n];
        recv_node_displs[n + 1] = recv_node_displs[n] + recv_node_size[n];
    }

    std::vector<char> global_send_buf((size_t)send_node_displs[num_nodes] * item_bytes);
    for (int n = 0; n < num_nodes; n++)
    {
        int off = send_node_displs[n];
        for (auto& c : send_by_node[n])
        {
            memcpy(&global_send_buf[(size_t)off * item_bytes],
                   &send_buffer[(size_t)c.displ * item_bytes],
                   (size_t)c.count * item_bytes);
            off += c.count;
        }
    }
    std::vector<char> global_recv_buf((size_t)recv_node_displs[num_nodes] * item_bytes);

    int global_tag;
    MPIL_Comm_tag(comm, &global_tag);

    std::vector<MPI_Request> global_reqs;
    global_reqs.reserve(2 * num_nodes);
    for (int n = 0; n < num_nodes; n++)
    {
        if (recv_node_size[n] == 0)
        {
            continue;
        }
        MPI_Request req;
        MPI_Irecv(&global_recv_buf[(size_t)recv_node_displs[n] * item_bytes],
                  recv_node_size[n] * item_bytes,
                  MPI_BYTE,
                  n,
                  global_tag,
                  comm->group_comm,
                  &req);
        global_reqs.push_back(req);
    }
    for (int n = 0; n < num_nodes; n++)
    {
        if (send_node_size[n] == 0)
        {
            continue;
        }
        MPI_Request req;
        MPI_Isend(&global_send_buf[(size_t)send_node_displs[n] * item_bytes],
                  send_node_size[n] * item_bytes,
                  MPI_BYTE,
                  n,
                  global_tag,
                  comm->group_comm,
                  &req);
        global_reqs.push_back(req);
    }
    if (!global_reqs.empty())
    {
        MPI_Waitall((int)global_reqs.size(), global_reqs.data(), MPI_STATUSES_IGNORE);
    }

    /**************************************************************
     * 4. Local_R step: disaggregate what I received above and
     *    forward each piece to the process on my node that needs
     *    it (comm->local_comm), again with known sizes/partners.
     **************************************************************/
    int local_tag;
    MPIL_Comm_tag(comm, &local_tag);

    std::vector<int> relay_recv_displs(ppn + 1, 0);
    for (int p = 0; p < ppn; p++)
    {
        relay_recv_displs[p + 1] = relay_recv_displs[p] + recv_from_relay_size[p];
    }
    std::vector<char> relay_recv_buf((size_t)relay_recv_displs[ppn] * item_bytes);

    std::vector<MPI_Request> local_reqs;
    local_reqs.reserve(2 * ppn);
    for (int p = 0; p < ppn; p++)
    {
        if (recv_from_relay_size[p] == 0)
        {
            continue;
        }
        MPI_Request req;
        MPI_Irecv(&relay_recv_buf[(size_t)relay_recv_displs[p] * item_bytes],
                  recv_from_relay_size[p] * item_bytes,
                  MPI_BYTE,
                  p,
                  local_tag,
                  comm->local_comm,
                  &req);
        local_reqs.push_back(req);
    }

    std::vector<int> local_send_displs(ppn + 1, 0);
    for (int p = 0; p < ppn; p++)
    {
        local_send_displs[p + 1] = local_send_displs[p] + local_send_size[p];
    }
    std::vector<char> local_send_buf((size_t)local_send_displs[ppn] * item_bytes);
    for (const Route& r : routes)
    {
        size_t src_off = (size_t)(recv_node_displs[r.source_node] + r.global_offset) * item_bytes;
        size_t dst_off = (size_t)(local_send_displs[r.local_q] + r.local_offset) * item_bytes;
        memcpy(&local_send_buf[dst_off], &global_recv_buf[src_off], (size_t)r.count * item_bytes);
    }

    for (int p = 0; p < ppn; p++)
    {
        if (local_send_size[p] == 0)
        {
            continue;
        }
        MPI_Request req;
        MPI_Isend(&local_send_buf[(size_t)local_send_displs[p] * item_bytes],
                  local_send_size[p] * item_bytes,
                  MPI_BYTE,
                  p,
                  local_tag,
                  comm->local_comm,
                  &req);
        local_reqs.push_back(req);
    }
    if (!local_reqs.empty())
    {
        MPI_Waitall((int)local_reqs.size(), local_reqs.data(), MPI_STATUSES_IGNORE);
    }

    // Split each relay's aggregated buffer back into the caller's recvbuf.
    for (int p = 0; p < ppn; p++)
    {
        int off = relay_recv_displs[p];
        for (auto& entry : recv_from_relay[p])
        {
            int idx = entry.second;
            int cnt = recvcounts[idx];
            memcpy(&recv_buffer[(size_t)rdispls[idx] * item_bytes],
                   &relay_recv_buf[(size_t)off * item_bytes],
                   (size_t)cnt * item_bytes);
            off += cnt;
        }
    }

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif
