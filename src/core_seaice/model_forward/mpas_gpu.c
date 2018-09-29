#include <openacc.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <cuda_profiler_api.h>

void start_profile()
{
    cudaProfilerStart();
}

void stop_profile()
{
    cudaProfilerStop();
}


int mpas_should_use_gpu(int rank)
{
    if ( NULL == getenv("CUDA_VISIBLE_DEVICES") ) return 0;
    
    MPI_Comm shmcomm;
    int err = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &shmcomm);
    
    int nranks, local;
    
    MPI_Comm_rank(shmcomm, &local);
    MPI_Comm_size(shmcomm, &nranks);
    // all ranks in shmcomm are on the same node
    char devices[64];
    
    int ssize = sizeof(devices);
    char * all_devices = malloc(nranks * ssize);
    
    
    strcpy(devices, getenv("CUDA_VISIBLE_DEVICES"));
    
    char * devn = strtok(devices, ",");
    int ndev = 0;
    while ( devn )
    {
        ++ndev;
        devn = strtok(NULL, ",");
    }
    
    MPI_Allgather(devices, ssize, MPI_CHAR,
                  all_devices, ssize, MPI_CHAR,
                  shmcomm);
    
    // first ndev lowest ranked procs that share the same set of GPUs
    // will be allowed to use the GPU.
    int should_use = 1;
    const char * check = all_devices;
    for ( int i = 0; i < local; ++i )
    {
        if ( strcmp(check, devices) == 0 )
        {
            --ndev;
        }
        check += ssize;
    }
    
    if ( ndev <= 0 ) should_use = 0;
    
    MPI_Comm_free(&shmcomm);
    free(all_devices);
    
    return should_use;
}


int mpas_select_gpu(int rank)
{
	int devsel = 0;
	int devlist[32];
	
	const char * devices = getenv("CUDA_VISIBLE_DEVICES");
    if ( devices )
    {
        //printf("rank %d: CUDA_VISIBLE_DEVICES = %s\n", rank, devices);
        
        char * devn = strtok(devices, ",");
        int ndev = 0;
        
        while ( devn )
        {
            devlist[ndev++] = atoi(devn);
            devn = strtok(NULL, ",");
        }
        
        if ( 0 == ndev )
        {
            ndev = acc_get_num_devices( acc_device_nvidia );
            for ( int i = 0; i < ndev; ++i )
            {
                devlist[i] = i;
            }
        }
        
        if ( ndev > 0 )
        {
            //devsel = devlist[rank % ndev];
            devsel = rank % ndev;
        }
    }
    else
    {
        int nranks;
        
        int hostid = gethostid();
        
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        
        int all_hosts[nranks];
        
        //fprintf(stderr, "rank %d: nranks = %d\n", rank, nranks);
        MPI_Allgather(&hostid, 1, MPI_INTEGER, all_hosts, 1, MPI_INTEGER, MPI_COMM_WORLD);
        //fprintf(stderr, "rank %d: allgather done.\n", rank);
        int nlocal = 0;
        int localprocs[nranks];
        for ( int i = 0; i < nranks; ++i )
        {
            localprocs[i] = 0;
            if ( hostid == all_hosts[i] )
            {
                localprocs[i] = nlocal;
                ++nlocal;
            }
        }
        
        int ndev = acc_get_num_devices(acc_device_nvidia);
        //fprintf(stderr, "rank %d: ndev = %d.\n", rank, ndev);
        if ( ndev < nlocal )
        {
            devsel = localprocs[rank] % ndev;
        }
        else
        {
            devsel = localprocs[rank];
        }
    }
	
    //printf("rank %d: selecting device %d\n", rank, devsel);

	return devsel;
}
