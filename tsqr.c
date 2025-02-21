#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>


#define N 4  // Number of columns in the matrix
#define BLOCK_ROWS 100  // Number of rows per processor (adjust as needed)

// Function to perform local QR factorization using LAPACK
void local_qr(double *A, int rows, int cols, double *Q, double *R) {
    int lda = cols;
    int lwork = cols * cols;
    double *work = (double *)malloc(lwork * sizeof(double));
    double *tau = (double *)malloc(cols * sizeof(double));

    // Perform QR factorization
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows, cols, A, lda, tau);
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, cols, cols, A, lda, tau);

    // Extract Q and R
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Q[i * cols + j] = A[i * cols + j];
            if (i == j) R[i * cols + j] = A[i * cols + j];
            else if (i < j) R[i * cols + j] = A[i * cols + j];
            else R[i * cols + j] = 0.0;
        }
    }

    free(work);
    free(tau);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            printf("This program requires exactly 4 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Each process holds a block of rows
    double *A_local = (double *)malloc(BLOCK_ROWS * N * sizeof(double));
    double *Q_local = (double *)malloc(BLOCK_ROWS * N * sizeof(double));
    double *R_local = (double *)malloc(N * N * sizeof(double));

    // Initialize local matrix (random data for testing)
    srand(rank + 1);
    for (int i = 0; i < BLOCK_ROWS; i++) {
        for (int j = 0; j < N; j++) {
            A_local[i * N + j] = (double)rand() / RAND_MAX;
        }
    }

    // Perform local QR factorization
    local_qr(A_local, BLOCK_ROWS, N, Q_local, R_local);

    // Gather all R matrices at the root process (rank 0)
    double *R_all = NULL;
    if (rank == 0) {
        R_all = (double *)malloc(size * N * N * sizeof(double));
    }
    MPI_Gather(R_local, N * N, MPI_DOUBLE, R_all, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Combine R matrices hierarchically at the root process
    if (rank == 0) {
        double *R_combined = (double *)malloc(2 * N * N * sizeof(double));
        double *R_final = (double *)malloc(N * N * sizeof(double));

        // Combine R1 and R2
        for (int i = 0; i < N * N; i++) {
            R_combined[i] = R_all[i];
            R_combined[N * N + i] = R_all[N * N + i];
        }
        local_qr(R_combined, 2 * N, N, NULL, R_final);

        // Combine R3 and R4
        for (int i = 0; i < N * N; i++) {
            R_combined[i] = R_all[2 * N * N + i];
            R_combined[N * N + i] = R_all[3 * N * N + i];
        }
        local_qr(R_combined, 2 * N, N, NULL, R_final);

        // Final R matrix
        printf("Final R matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", R_final[i * N + j]);
            }
            printf("\n");
        }

        free(R_all);
        free(R_combined);
        free(R_final);
    }

    free(A_local);
    free(Q_local);
    free(R_local);

    MPI_Finalize();
    return 0;
}