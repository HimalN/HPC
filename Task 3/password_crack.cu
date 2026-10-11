/*
 * Student Name: Himal Perera
 * Module      : 6CS005 - High Performance Computing
 * Task        : Task 3 - Password Cracking with Files using CUDA
 *
 * COMPILE: nvcc -O3 password_crack.cu -o password_crack
 * RUN    : ./password_crack passwords.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define ENCRYPTED_LEN 11 // 10 chars + '\0'
#define RAW_LEN 5        // 4 chars + '\0'

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__device__ void cudaCrypt(const char* rawPassword, char* newPassword) {
    // Note: The 'static' keyword was removed from newPassword to guarantee
    // thread-isolation across thousands of concurrent GPU threads.
    newPassword[0] = rawPassword[0] + 2;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 3;
    newPassword[4] = rawPassword[1] - 3;
    newPassword[5] = rawPassword[1] - 1;
    newPassword[6] = rawPassword[2] + 2;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 4;
    newPassword[10] = '\0';

    for (int i = 0; i < 10; i++) {
        if (i >= 0 && i < 6) { // checking all lower case letter limits
            if (newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if (newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else { // checking number section
            if (newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if (newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
}

__global__ void crackPasswords(char *d_encrypted, char *d_decrypted, int num_passwords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_passwords) return;

    char* target = &d_encrypted[idx * ENCRYPTED_LEN];
    char* result = &d_decrypted[idx * RAW_LEN];

    char rawTest[RAW_LEN];
    char encryptedTest[ENCRYPTED_LEN];
    rawTest[4] = '\0';

    for (char l1 = 'a'; l1 <= 'z'; l1++) {
        for (char l2 = 'a'; l2 <= 'z'; l2++) {
            for (char n1 = '0'; n1 <= '9'; n1++) {
                for (char n2 = '0'; n2 <= '9'; n2++) {

                    rawTest[0] = l1; rawTest[1] = l2; rawTest[2] = n1; rawTest[3] = n2;

                    // Generate the 10-char hash using the __device__ function
                    cudaCrypt(rawTest, encryptedTest);

                    // Custom device-safe string comparison
                    bool match = true;
                    for (int c = 0; c < 10; c++) {
                        if (target[c] != encryptedTest[c]) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
                        result[0] = rawTest[0];
                        result[1] = rawTest[1];
                        result[2] = rawTest[2];
                        result[3] = rawTest[3];
                        result[4] = '\0';
                        return;
                    }
                }
            }
        }
    }

    result[0] = '?'; result[1] = '\0';
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <encrypted_passwords.txt>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *input_filename = argv[1];

    FILE *infile = fopen(input_filename, "r");
    if (!infile) {
        fprintf(stderr, "Error: Could not open file %s\n", input_filename);
        return EXIT_FAILURE;
    }

    int capacity = 1000;
    int num_passwords = 0;

    char *h_encrypted = (char *)malloc(capacity * ENCRYPTED_LEN * sizeof(char));
    char buffer[ENCRYPTED_LEN + 2]; // Accommodate \n and \0

    while (fscanf(infile, "%10s", buffer) == 1) {
        if (num_passwords >= capacity) {
            capacity *= 2;
            h_encrypted = (char *)realloc(h_encrypted, capacity * ENCRYPTED_LEN * sizeof(char));
        }
        strcpy(&h_encrypted[num_passwords * ENCRYPTED_LEN], buffer);
        num_passwords++;
    }
    fclose(infile);

    if (num_passwords == 0) {
        printf("The input file is empty.\n");
        free(h_encrypted);
        return EXIT_SUCCESS;
    }

    char *h_decrypted = (char *)malloc(num_passwords * RAW_LEN * sizeof(char));

    char *d_encrypted = NULL;
    char *d_decrypted = NULL;

    size_t encrypted_size = num_passwords * ENCRYPTED_LEN * sizeof(char);
    size_t decrypted_size = num_passwords * RAW_LEN * sizeof(char);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_encrypted, encrypted_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_decrypted, decrypted_size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_encrypted, h_encrypted, encrypted_size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_passwords + threadsPerBlock - 1) / threadsPerBlock;

    printf("Cracking %d passwords using %d blocks and %d threads per block...\n", num_passwords, blocksPerGrid, threadsPerBlock);

    crackPasswords<<<blocksPerGrid, threadsPerBlock>>>(d_encrypted, d_decrypted, num_passwords);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_decrypted, d_decrypted, decrypted_size, cudaMemcpyDeviceToHost));

    FILE *outfile = fopen("decrypted.txt", "w");
    if (!outfile) {
        fprintf(stderr, "Error: Could not create decrypted.txt\n");
    } else {
        for (int i = 0; i < num_passwords; i++) {
            fprintf(outfile, "%s\n", &h_decrypted[i * RAW_LEN]);
        }
        fclose(outfile);
        printf("Successfully saved results to decrypted.txt\n");
    }

    cudaFree(d_encrypted);
    cudaFree(d_decrypted);
    free(h_encrypted);
    free(h_decrypted);

    return EXIT_SUCCESS;
}
