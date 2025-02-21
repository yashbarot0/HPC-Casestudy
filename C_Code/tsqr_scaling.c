#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <time.h>

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

    // Parameters for scaling tests
    int m_values[] = {100, 500, 1000, 5000, 10000};  // Vary m
    int n_values[] = {4, 8, 16, 32, 64};                     // Vary n
    int num_m = sizeof(m_values) / sizeof(m_values[0]);
    int num_n = sizeof(n_values) / sizeof(n_values[0]);

    if (rank == 0) {
        printf("Scaling with respect to m (n fixed):\n");
        printf("m\tTime (s)\n");
    }

    // Scaling with respect to m (n fixed)
    int n_fixed = 4;
    for (int i = 0; i < num_m; i++) {
        int m = m_values[i];
        int block_rows = m / size;

        // Allocate memory for local data
        double *A_local = (double *)malloc(block_rows * n_fixed * sizeof(double));
        double *Q_local = (double *)malloc(block_rows * n_fixed * sizeof(double));
        double *R_local = (double *)malloc(n_fixed * n_fixed * sizeof(double));

        if (A_local == NULL || Q_local == NULL || R_local == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize local matrix with random data
        srand(rank + 1);
        for (int i = 0; i < block_rows * n_fixed; i++) A_local[i] = (double)rand() / RAND_MAX;

        // Measure execution time
        double start_time = MPI_Wtime();

        // Perform local QR
        local_qr(A_local, block_rows, n_fixed, Q_local, R_local);

        // Gather all R matrices at root (rank 0)
        double *R_all = (double *)malloc(size * n_fixed * n_fixed * sizeof(double));
        if (R_all == NULL) {
            fprintf(stderr, "Memory allocation failed for R_all\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Gather(R_local, n_fixed * n_fixed, MPI_DOUBLE, R_all, n_fixed * n_fixed, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Root process combines R matrices hierarchically
        if (rank == 0) {
            double *R_combined = (double *)malloc(2 * n_fixed * n_fixed * sizeof(double));
            double *R_temp = (double *)malloc(n_fixed * n_fixed * sizeof(double));
            double *R_final = (double *)malloc(n_fixed * n_fixed * sizeof(double));

            if (R_combined == NULL || R_temp == NULL || R_final == NULL) {
                fprintf(stderr, "Memory allocation failed at root\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Combine R1 and R2
            for (int i = 0; i < n_fixed * n_fixed; i++) {
                R_combined[i] = R_all[i];               // R1
                R_combined[n_fixed * n_fixed + i] = R_all[n_fixed * n_fixed + i]; // R2
            }
            local_qr(R_combined, 2 * n_fixed, n_fixed, NULL, R_temp);

            // Combine R3 and R4
            for (int i = 0; i < n_fixed * n_fixed; i++) {
                R_combined[i] = R_all[2 * n_fixed * n_fixed + i];   // R3
                R_combined[n_fixed * n_fixed + i] = R_all[3 * n_fixed * n_fixed + i]; // R4
            }
            local_qr(R_combined, 2 * n_fixed, n_fixed, NULL, R_final);

            // Combine R_temp and R_final
            for (int i = 0; i < n_fixed * n_fixed; i++) R_combined[i] = R_temp[i];
            for (int i = 0; i < n_fixed * n_fixed; i++) R_combined[n_fixed * n_fixed + i] = R_final[i];
            local_qr(R_combined, 2 * n_fixed, n_fixed, NULL, R_final);

            // Measure end time
            double end_time = MPI_Wtime();
            printf("%d\t%.6f\n", m, end_time - start_time);

            free(R_combined);
            free(R_temp);
            free(R_final);
        }

        free(A_local);
        free(Q_local);
        free(R_local);
        free(R_all);
    }

    if (rank == 0) {
        printf("\nScaling with respect to n (m fixed):\n");
        printf("n\tTime (s)\n");
    }

    // Scaling with respect to n (m fixed)
    int m_fixed = 1000;
    for (int i = 0; i < num_n; i++) {
        int n = n_values[i];
        int block_rows = m_fixed / size;

        // Allocate memory for local data
        double *A_local = (double *)malloc(block_rows * n * sizeof(double));
        double *Q_local = (double *)malloc(block_rows * n * sizeof(double));
        double *R_local = (double *)malloc(n * n * sizeof(double));

        if (A_local == NULL || Q_local == NULL || R_local == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize local matrix with random data
        srand(rank + 1);
        for (int i = 0; i < block_rows * n; i++) A_local[i] = (double)rand() / RAND_MAX;

        // Measure execution time
        double start_time = MPI_Wtime();

        // Perform local QR
        local_qr(A_local, block_rows, n, Q_local, R_local);

        // Gather all R matrices at root (rank 0)
        double *R_all = (double *)malloc(size * n * n * sizeof(double));
        if (R_all == NULL) {
            fprintf(stderr, "Memory allocation failed for R_all\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Gather(R_local, n * n, MPI_DOUBLE, R_all, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Root process combines R matrices hierarchically
        if (rank == 0) {
            double *R_combined = (double *)malloc(2 * n * n * sizeof(double));
            double *R_temp = (double *)malloc(n * n * sizeof(double));
            double *R_final = (double *)malloc(n * n * sizeof(double));

            if (R_combined == NULL || R_temp == NULL || R_final == NULL) {
                fprintf(stderr, "Memory allocation failed at root\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Combine R1 and R2
            for (int i = 0; i < n * n; i++) {
                R_combined[i] = R_all[i];               // R1
                R_combined[n * n + i] = R_all[n * n + i]; // R2
            }
            local_qr(R_combined, 2 * n, n, NULL, R_temp);

            // Combine R3 and R4
            for (int i = 0; i < n * n; i++) {
                R_combined[i] = R_all[2 * n * n + i];   // R3
                R_combined[n * n + i] = R_all[3 * n * n + i]; // R4
            }
            local_qr(R_combined, 2 * n, n, NULL, R_final);

            // Combine R_temp and R_final
            for (int i = 0; i < n * n; i++) R_combined[i] = R_temp[i];
            for (int i = 0; i < n * n; i++) R_combined[n * n + i] = R_final[i];
            local_qr(R_combined, 2 * n, n, NULL, R_final);

            // Measure end time
            double end_time = MPI_Wtime();
            printf("%d\t%.6f\n", n, end_time - start_time);

            free(R_combined);
            free(R_temp);
            free(R_final);
        }

        free(A_local);
        free(Q_local);
        free(R_local);
        free(R_all);
    }

    MPI_Finalize();
    return 0;
}