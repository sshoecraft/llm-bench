#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    void *ptr0, *ptr1;
    cudaSetDevice(0);
    cudaMalloc(&ptr0, 1024);
    printf("GPU 0 allocation: %p\n", ptr0);
    
    cudaSetDevice(1);
    cudaMalloc(&ptr1, 1024);
    printf("GPU 1 allocation: %p\n", ptr1);
    
    cudaFree(ptr0);
    cudaFree(ptr1);
    return 0;
}
