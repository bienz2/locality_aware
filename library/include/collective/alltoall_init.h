#ifndef MPI_ADVANCE_ALLTOALL_INIT_H
#define MPI_ADVANCE_ALLTOALL_INIT_H

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "persistent/MPIL_Request.h"
#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"

#ifdef __cplusplus
extern "C" {
#endif

int alltoall_rma_start(MPIL_Request* request);
int alltoall_rma_wait(MPIL_Request* request, MPI_Status* status);
int alltoall_nonblocking_start(MPIL_Request* request);
int alltoall_nonblocking_wait(MPIL_Request* request, MPI_Status* status);
int alltoall_pairwise_start(MPIL_Request* request);
int alltoall_pairwise_wait(MPIL_Request* request, MPI_Status* status);

/** @brief Function pointer to alltoall init implemenation
 * @details
 * Uses the parameters of standard MPI_Alltoall API, except replacing MPI_Comm with
 * MPIL_Comm most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API alltoall switch statement targets one of these.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/
typedef int (*alltoall_init_ftn)(
    const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPIL_Comm*,
    MPIL_Info* info, MPIL_Request** req_ptr);

/** @brief Function pointer to alltoall init helper function..
 * @details
 * Uses the parameters of standard MPI_Alltoall API, plus a tag for additional options.
 * usually invoked by a function of type alltoall_ftn.
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] tag unique tag for matching messages.
 * @param [in] info MPIL_Info*
 * @param [out] req_ptr MPIL_Request**
 **/
typedef int (*alltoall_init_helper_ftn)(const void*,
                                   const int,
                                   MPI_Datatype,
                                   void*,
                                   const int,
                                   MPI_Datatype,
                                   MPI_Comm,
                                   int tag,
                                   MPIL_Info* info, 
                                   MPIL_Request** req_ptr);

//** External Wrappers
//**//----------------------------------------------------------------------
/** @brief Call the rma implementation.
 * @details calls get_tag() then call pairwise_helper() with the same input parameters
 * plus the found tag.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info*
 * @param [out] req_ptr MPIL_Request**
 * @return returns value of the pairwise_helper call.
 */
int alltoall_rma_init(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm,
                      MPIL_Info* info, 
                      MPIL_Request** req_ptr);

/** @brief Call the pairwise implementation.
 * @details calls get_tag() then call pairwise_helper() with the same input parameters
 * plus the found tag.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info*
 * @param [out] req_ptr MPIL_Request**
 * @return returns value of the pairwise_helper call.
 */
int alltoall_pairwise_init(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm,
                      MPIL_Info* info, 
                      MPIL_Request** req_ptr);

/** @brief Call the nonblocking implementation.
 * @details calls get_tag() then call pairwise_helper() with the same input parameters
 * plus the found tag.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info*
 * @param [out] req_ptr MPIL_Request**
 * @return returns value of the pairwise_helper call.
 */
int alltoall_nonblocking_init(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm,
                      MPIL_Info* info, 
                      MPIL_Request** req_ptr);

//** Intermediate Wrappers
//**//----------------------------------------------------------------------
int alltoall_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIL_Comm* xcomm,
        MPIL_Info* xinfo,
        MPIL_Request** request_ptr);

#ifdef __cplusplus
}
#endif

#endif

