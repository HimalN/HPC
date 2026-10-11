/*
 * Student Name: Himal Perera
 * Module      : 6CS005 - High Performance Computing
 * Task        : Task 2 - Multiple Operations with Matrices using Multithreading (OpenMP)
 *
 * COMPILE: gcc -O3 -fopenmp matrix_ops.c -o matrix_ops -lm
 * RUN    : ./matrix_ops MatData.txt 4
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <omp.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

static int allocate_matrix(Matrix *m, int rows, int cols) {
    m->rows = rows;
    m->cols = cols;
    m->data = (double **)calloc(rows, sizeof(double *));
    if (!m->data) return 0;
    for (int i = 0; i < rows; i++) {
        m->data[i] = (double *)malloc(cols * sizeof(double));
        if (!m->data[i]) {
            for (int k = 0; k < i; k++) free(m->data[k]);
            free(m->data);
            m->data = NULL;
            return 0;
        }
    }
    return 1;
}

static void free_matrix(Matrix *m) {
    if (m->data) {
        for (int i = 0; i < m->rows; i++) free(m->data[i]);
        free(m->data);
        m->data = NULL;
    }
}

static void skip_blanks(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) == ' ' || c == '\t') { }
    if (c != EOF) ungetc(c, fp);
}

static int at_line_end(FILE *fp) {
    skip_blanks(fp);
    int c = fgetc(fp);
    if (c == '\r') c = fgetc(fp);
    if (c == '\n' || c == EOF) return 1;
    ungetc(c, fp);
    return 0;
}

static int read_matrix(FILE *fp, Matrix *m, int index) {
    int c;
    while ((c = fgetc(fp)) != EOF && isspace(c)) { }
    if (c == EOF) return 0;
    ungetc(c, fp);

    int r, cl;
    if (fscanf(fp, "%d,%d", &r, &cl) != 2 || !at_line_end(fp)) {
        fprintf(stderr, "Error: matrix %d has an invalid header (expected rows,cols).\n", index);
        return -1;
    }
    if (r <= 0 || cl <= 0) {
        fprintf(stderr, "Error: matrix %d has invalid dimensions (%d,%d).\n", index, r, cl);
        return -1;
    }
    if (!allocate_matrix(m, r, cl)) {
        fprintf(stderr, "Error: out of memory allocating matrix %d (%dx%d).\n", index, r, cl);
        return -1;
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < cl; j++) {
            skip_blanks(fp);
            c = fgetc(fp);
            if (c == EOF || c == '\n' || c == '\r') {
                fprintf(stderr, "Error: matrix %d, row %d has %d value(s), expected %d "
                                "(or the file ended early).\n", index, i + 1, j, cl);
                free_matrix(m);
                return -1;
            }
            ungetc(c, fp);

            if (fscanf(fp, "%lf", &m->data[i][j]) != 1) {
                fprintf(stderr, "Error: matrix %d, row %d, column %d is not a number.\n",
                        index, i + 1, j + 1);
                free_matrix(m);
                return -1;
            }

            skip_blanks(fp);
            c = fgetc(fp);
            if (j < cl - 1) {
                if (c != ',') {
                    fprintf(stderr, "Error: matrix %d, row %d: expected ',' after column %d "
                                    "(row too short or bad separator).\n", index, i + 1, j + 1);
                    free_matrix(m);
                    return -1;
                }
            } else {
                if (c == '\r') c = fgetc(fp);
                if (c != '\n' && c != EOF) {
                    fprintf(stderr, "Error: matrix %d, row %d has more than %d values.\n",
                            index, i + 1, cl);
                    free_matrix(m);
                    return -1;
                }
            }
        }
    }
    return 1;
}

static void print_matrix(FILE *fp, const Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (isnan(m->data[i][j])) fprintf(fp, "NaN");
            else                       fprintf(fp, "%.6f", m->data[i][j]);
            if (j < m->cols - 1) fputc(',', fp);
        }
        fputc('\n', fp);
    }
}

static int cap_threads(int requested, int limit) {
    int t = (requested > limit) ? limit : requested;
    return (t < 1) ? 1 : t;
}

static void elementwise(const Matrix *A, const Matrix *B, FILE *fp, int req,
                        char op, const char *name) {
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(fp, "%s cannot be done (shapes differ).\n", name);
        return;
    }
    Matrix C;
    if (!allocate_matrix(&C, A->rows, A->cols)) {
        fprintf(fp, "%s cannot be done (out of memory).\n", name);
        return;
    }
    int threads = cap_threads(req, A->rows);

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            double a = A->data[i][j], b = B->data[i][j];
            switch (op) {
                case '+': C.data[i][j] = a + b; break;
                case '-': C.data[i][j] = a - b; break;
                case '*': C.data[i][j] = a * b; break;
                default:  C.data[i][j] = (b == 0.0) ? NAN : a / b; break;
            }
        }
    }

    fprintf(fp, "%s - %d,%d\n", name, C.rows, C.cols);
    print_matrix(fp, &C);
    free_matrix(&C);
}

static void transpose_matrix(const Matrix *M, const char *label, FILE *fp, int req) {
    Matrix T;
    if (!allocate_matrix(&T, M->cols, M->rows)) {
        fprintf(fp, "Transpose of %s cannot be done (out of memory).\n", label);
        return;
    }
    int threads = cap_threads(req, T.rows);

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < T.rows; i++)
        for (int j = 0; j < T.cols; j++)
            T.data[i][j] = M->data[j][i];

    fprintf(fp, "Transpose of %s - %d,%d\n", label, T.rows, T.cols);
    print_matrix(fp, &T);
    free_matrix(&T);
}

static void mult_matrices(const Matrix *A, const Matrix *B, FILE *fp, int req) {
    if (A->cols != B->rows) {
        fprintf(fp, "Matrix multiplication cannot be done (A.cols != B.rows).\n");
        return;
    }
    Matrix C;
    if (!allocate_matrix(&C, A->rows, B->cols)) {
        fprintf(fp, "Matrix multiplication cannot be done (out of memory).\n");
        return;
    }
    int threads = cap_threads(req, C.rows);

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++)
                sum += A->data[i][k] * B->data[k][j];
            C.data[i][j] = sum;
        }
    }

    fprintf(fp, "Matrix multiplication - %d,%d\n", C.rows, C.cols);
    print_matrix(fp, &C);
    free_matrix(&C);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <num_threads>\n", argv[0]);
        return 1;
    }

    char *end = NULL;
    long t = strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || t < 1 || t > 100000) {
        fprintf(stderr, "Error: num_threads must be a positive integer.\n");
        return 1;
    }
    int req_threads = (int)t;

    FILE *in_fp = fopen(argv[1], "r");
    if (!in_fp) {
        fprintf(stderr, "Error: could not open input file '%s'.\n", argv[1]);
        return 1;
    }
    FILE *out_fp = fopen("results.txt", "w");
    if (!out_fp) {
        fprintf(stderr, "Error: could not open results.txt for writing.\n");
        fclose(in_fp);
        return 1;
    }

    Matrix A = {0, 0, NULL}, B = {0, 0, NULL};
    int pair = 1, status = 0;

    while (1) {
        int sa = read_matrix(in_fp, &A, 2 * pair - 1);
        if (sa == 0) break;
        if (sa < 0) { status = 1; break; }

        int sb = read_matrix(in_fp, &B, 2 * pair);
        if (sb == 0) {
            fprintf(stderr, "Error: matrix %d has no partner (odd number of matrices).\n",
                    2 * pair - 1);
            free_matrix(&A);
            status = 1;
            break;
        }
        if (sb < 0) { free_matrix(&A); status = 1; break; }

        fprintf(out_fp, "--- Pair %d Results ---\n", pair);
        elementwise(&A, &B, out_fp, req_threads, '+', "Addition");
        fputc('\n', out_fp);
        elementwise(&A, &B, out_fp, req_threads, '-', "Subtraction");
        fputc('\n', out_fp);
        elementwise(&A, &B, out_fp, req_threads, '*', "Element-wise multiplication");
        fputc('\n', out_fp);
        elementwise(&A, &B, out_fp, req_threads, '/', "Element-wise division");
        fputc('\n', out_fp);
        transpose_matrix(&A, "A", out_fp, req_threads);
        fputc('\n', out_fp);
        transpose_matrix(&B, "B", out_fp, req_threads);
        fputc('\n', out_fp);
        mult_matrices(&A, &B, out_fp, req_threads);
        fputs("\n\n", out_fp);

        free_matrix(&A);
        free_matrix(&B);
        pair++;
    }

    fclose(in_fp);
    fclose(out_fp);

    if (status == 0)
        printf("Done. %d pair(s) processed. Results in results.txt\n", pair - 1);
    else
        fprintf(stderr, "Stopped because of an input error. %d pair(s) completed before it.\n",
                pair - 1);
    return status;
}
