
#include <stdio.h> /* printf */
#include <stdlib.h> /* atoi, NULL, malloc, free */
#include <mpi.h> /* MPI_* */

#define N_TRY 200
#define MAX_COUNT (256*1024)

/* command line arguments */
int root = 0;      /* reduction target */
int vect_size = 1; /* vector size */

/* all-to-one vector reduce */
int MPX_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
}


int main(int argc, char *argv[])
{
	double start, finish, runtime, lowtime;
	int i, vsize, tcnt, ecnt;
	int ret; /* function return value */
	int cmsz; /* number of processes in communicator group */
	int rank; /* process rank (index starting from zero) */
	int *sendbuf;
	int *recvbuf;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &cmsz);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc > 1) root = atoi(argv[1]);
	if (argc > 2) vect_size = atoi(argv[2]);
	if ((sendbuf = (int *)malloc(sizeof(int)*vect_size)) == NULL) {
		printf(" -- error rank:%d malloc(sendbuf)\n", rank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	if ((recvbuf = (int *)malloc(sizeof(int)*vect_size)) == NULL) {
		printf(" -- error rank:%d malloc(recvbuf)\n", rank);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	for (i = 0; i < vect_size; i++) sendbuf[i] = i+1;

	if (rank == root) {
		printf("procs:%d root:%d\nvsize   time(usec)\n", cmsz, root);
	}
	ecnt = 0;
	for (vsize = 1; vsize <= vect_size; vsize <<= 1) {
		/* benchmark the function */
		lowtime = 1e10;
		for (tcnt = 0; tcnt < N_TRY; tcnt++) {
			start = MPI_Wtime();
			ret = MPI_Reduce(sendbuf, recvbuf, vsize, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
			finish = MPI_Wtime();
			if (ret != MPI_SUCCESS) {
				static char str[MPI_MAX_ERROR_STRING];
				int slen;
				MPI_Error_string(ret, str, &slen);
				printf(" -- error rank:%d %s\n", rank, str);
				MPI_Abort(MPI_COMM_WORLD, ret); /* or break; */
			}
			runtime = (finish - start);
			if (runtime < lowtime) lowtime = runtime;
		}

		/* only on root */
		if (rank == root) {
			/* verify results */
			for (i = 0; i < vsize && ecnt < 5; i++) {
				if (recvbuf[i] != (i+1)*cmsz) {
					printf(" -- error recvbuf[%d]:%d\n", i, recvbuf[i]);
					ecnt++;
					break;
				}
			}

			printf("%d\t%f\n", vsize, lowtime * 1e6);
		}
	}
	free(sendbuf);
	free(recvbuf);

	MPI_Finalize();
}
