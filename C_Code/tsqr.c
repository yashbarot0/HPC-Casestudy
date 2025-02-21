#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>

#define N 4        // Number of columns
#define BLOCK_ROWS 100  // Rows per process

// Local QR factorization using LAPACK
void local_qr(double *A, int rows, int cols, double *Q, double *R) {
    int lda = cols;
    double *tau = (double *)malloc(cols * sizeof(double));
    if (tau == NULL) {
        fprintf(stderr, "Memory allocation failed for tau\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute QR (A is overwritten with Householder vectors)
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows, cols, A, lda, tau);

    // Extract R matrix (upper triangular)
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            if (j >= i) R[i * cols + j] = A[i * cols + j];
            else R[i * cols + j] = 0.0;
        }
    }

    // Compute Q matrix if needed
    if (Q != NULL) {
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, cols, cols, A, lda, tau);
        for (int i = 0; i < rows * cols; i++) Q[i] = A[i];
    }

    free(tau);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) printf("Error: Use 4 processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Allocate memory for local data
    double *A_local = (double *)malloc(BLOCK_ROWS * N * sizeof(double));
    double *Q_local = (double *)malloc(BLOCK_ROWS * N * sizeof(double));
    double *R_local = (double *)malloc(N * N * sizeof(double));

    if (A_local == NULL || Q_local == NULL || R_local == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize local matrix with random data
    srand(rank + 1);
    for (int i = 0; i < BLOCK_ROWS * N; i++) A_local[i] = (double)rand() / RAND_MAX;

    // Perform local QR
    local_qr(A_local, BLOCK_ROWS, N, Q_local, R_local);

    // Gather all R matrices at root (rank 0)
    double *R_all = (double *)malloc(4 * N * N * sizeof(double));
    if (R_all == NULL) {
        fprintf(stderr, "Memory allocation failed for R_all\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Gather(R_local, N * N, MPI_DOUBLE, R_all, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root process combines R matrices hierarchically
    if (rank == 0) {
        double *R_combined = (double *)malloc(2 * N * N * sizeof(double));
        double *R_temp = (double *)malloc(N * N * sizeof(double));
        double *R_final = (double *)malloc(N * N * sizeof(double));

        if (R_combined == NULL || R_temp == NULL || R_final == NULL) {
            fprintf(stderr, "Memory allocation failed at root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Combine R1 and R2
        for (int i = 0; i < N * N; i++) {
            R_combined[i] = R_all[i];               // R1
            R_combined[N * N + i] = R_all[N * N + i]; // R2
        }
        local_qr(R_combined, 2 * N, N, NULL, R_temp);

        // Combine R3 and R4
        for (int i = 0; i < N * N; i++) {
            R_combined[i] = R_all[2 * N * N + i];   // R3
            R_combined[N * N + i] = R_all[3 * N * N + i]; // R4
        }
        local_qr(R_combined, 2 * N, N, NULL, R_final);

        // Combine R_temp and R_final
        for (int i = 0; i < N * N; i++) R_combined[i] = R_temp[i];
        for (int i = 0; i < N * N; i++) R_combined[N * N + i] = R_final[i];
        local_qr(R_combined, 2 * N, N, NULL, R_final);

        // Print final R
        printf("Final R:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) printf("%10.6f ", R_final[i * N + j]);
            printf("\n");
        }

        free(R_combined);
        free(R_temp);
        free(R_final);
    }

    // Free all memory
    free(A_local);
    free(Q_local);
    free(R_local);
    free(R_all);

    MPI_Finalize();
    return 0;
}
