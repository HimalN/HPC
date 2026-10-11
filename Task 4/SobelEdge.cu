/*
 * Student Name: Himal Perera
 * Module      : 6CS005 - High Performance Computing
 * Task        : Task 4 - Sobel Edge Detection using CUDA across Many PNG Images
 *
 * COMPILE: nvcc -o SobelEdge sobel_edge.cu lodepng.cpp
 * RUN    : ./SobelEdge *.png
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lodepng.h"

#define BLOCK_DIM 16

/* Report a CUDA error, set the status and jump to the clean-up code. */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err_));           \
            status = 1;                                                      \
            goto cleanup;                                                    \
        }                                                                    \
    } while (0)

/* Grey value of pixel (x, y); 0 outside the image = zero padding. */
__device__ float greyAt(const unsigned char *img, int x, int y, int width, int height)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        return 0.0f;
    }

    size_t i = ((size_t)y * width + x) * 4;
    return 0.299f * img[i] + 0.587f * img[i + 1] + 0.114f * img[i + 2];
}

__global__ void sobelEdge(const unsigned char *inImg, unsigned char *outImg,
                          unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (int)width || y >= (int)height)    /* spare threads in edge blocks */
    {
        return;
    }

    const int gxKernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    const int gyKernel[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    float Gx = 0.0f;
    float Gy = 0.0f;

    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            float grey = greyAt(inImg, x + kx, y + ky, (int)width, (int)height);

            Gx += grey * gxKernel[ky + 1][kx + 1];
            Gy += grey * gyKernel[ky + 1][kx + 1];
        }
    }

    float magnitude = sqrtf(Gx * Gx + Gy * Gy);
    if (magnitude > 255.0f)
    {
        magnitude = 255.0f;
    }

    unsigned char edge = (unsigned char)magnitude;
    size_t index = ((size_t)y * width + x) * 4;

    outImg[index]     = edge;
    outImg[index + 1] = edge;
    outImg[index + 2] = edge;
    outImg[index + 3] = 255;
}

/* Process one image. Returns 0 on success, 1 on any error. */
static int processImage(const char *inFileName, const char *outFileName)
{
    int status = 0;
    unsigned char *c_inImg = NULL;
    unsigned char *c_outImg = NULL;
    unsigned char *d_inImg = NULL;
    unsigned char *d_outImg = NULL;
    unsigned int c_width = 0;
    unsigned int c_height = 0;
    unsigned int lodepngError = 0;
    size_t imgSize = 0;
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize(1, 1);

    /* 1. Read the image into CPU memory */
    lodepngError = lodepng_decode32_file(&c_inImg, &c_width, &c_height, inFileName);
    if (lodepngError)
    {
        fprintf(stderr, "Error decoding '%s': %u %s\n",
                inFileName, lodepngError, lodepng_error_text(lodepngError));
        return 1;
    }
    if (c_width == 0 || c_height == 0)
    {
        fprintf(stderr, "Error: '%s' has no pixels.\n", inFileName);
        free(c_inImg);
        return 1;
    }

    imgSize = (size_t)4 * c_width * c_height;       /* R G B A per pixel */

    c_outImg = (unsigned char *)malloc(imgSize);
    if (c_outImg == NULL)
    {
        fprintf(stderr, "Error: out of memory for '%s'.\n", inFileName);
        free(c_inImg);
        return 1;
    }

    /* 2. Allocate GPU memory from the image size and copy the image over */
    CUDA_CHECK(cudaMalloc((void **)&d_inImg, imgSize));
    CUDA_CHECK(cudaMalloc((void **)&d_outImg, imgSize));
    CUDA_CHECK(cudaMemcpy(d_inImg, c_inImg, imgSize, cudaMemcpyHostToDevice));

    /* 3. 2D grid sized from the image: one thread per pixel */
    gridSize.x = (c_width  + blockSize.x - 1) / blockSize.x;
    gridSize.y = (c_height + blockSize.y - 1) / blockSize.y;

    sobelEdge<<<gridSize, blockSize>>>(d_inImg, d_outImg, c_width, c_height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 4. Copy the edge image back to the host */
    CUDA_CHECK(cudaMemcpy(c_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost));

    /* 5. Write the edge-detected image */
    lodepngError = lodepng_encode32_file(outFileName, c_outImg, c_width, c_height);
    if (lodepngError)
    {
        fprintf(stderr, "Error encoding '%s': %u %s\n",
                outFileName, lodepngError, lodepng_error_text(lodepngError));
        status = 1;
    }
    else
    {
        printf("%s (%ux%u) -> %s\n", inFileName, c_width, c_height, outFileName);
    }

cleanup:
    /* 6. Free GPU and CPU memory for this image */
    cudaFree(d_inImg);
    cudaFree(d_outImg);
    free(c_inImg);
    free(c_outImg);

    return status;
}

int main(int argc, char *argv[])
{
    int failed = 0;
    int done = 0;
    int skipped = 0;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <image1.png> [image2.png ...]\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++)
    {
        const char *slash = strrchr(argv[i], '/');
        const char *baseName = (slash != NULL) ? slash + 1 : argv[i];
        char outFileName[1024];

        if (strncmp(baseName, "edge_", 5) == 0)
        {
            skipped++;                              /* an earlier output */
            continue;
        }

        int written = snprintf(outFileName, sizeof(outFileName), "edge_%s", baseName);
        if (written < 0 || (size_t)written >= sizeof(outFileName))
        {
            fprintf(stderr, "Error: file name too long: %s\n", argv[i]);
            failed++;
            continue;
        }

        if (processImage(argv[i], outFileName) == 0)
        {
            done++;
        }
        else
        {
            failed++;
        }
    }

    printf("Finished: %d image(s) processed, %d failed, %d skipped.\n", done, failed, skipped);
    return (failed > 0) ? 1 : 0;
}
