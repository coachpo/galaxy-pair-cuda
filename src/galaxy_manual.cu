#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define totaldegrees 90
#define binsperdegree 4
#define threadsperblock 256
// srun -p gpu -n 1 -t 10:00 --mem=1G -e err.txt -o out.txt time ./galaxy data_100k_arcmin.dat flat_100k_arcmin.dat result.out
// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
float *ra_real_dev, *decl_real_dev;

// number of real galaxies
int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
float *ra_sim_dev, *decl_sim_dev;
// number of simulated random galaxies
int NoofSim;

float *omega;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *histogramDR_dev, *histogramDD_dev, *histogramRR_dev;
unsigned int *d_histogram;

const int totalBins = totaldegrees * binsperdegree;
const float oneRad = 180.0f / M_PI;

int noofblocks;

__global__ void FillHistogram(unsigned int *histogram, int NoofReal, float *ra1, float *decl1, int NoofSim, float *ra2, float *decl2)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int shared_histogram[totalBins];

    int id = threadIdx.x;
    while (id < totalBins)
    {
        shared_histogram[id] = 0U;
        id += threadsperblock;
    }

    // wait for all threads to finish
    __syncthreads();

    if (threadId < NoofReal)
    {
        float cosDecl1 = cosf(decl1[threadId]);
        float sinDecl1 = sinf(decl1[threadId]);
        float ra1Val = ra1[threadId];
        float degree = 0.0;

        for (int i = 0; i < NoofReal; i += threadsperblock)
        {

            for (int j = 0; j < min(threadsperblock, NoofReal - i); j++)
            {
                float cosAngle = cosDecl1 * cosf(decl2[i + j]) * cosf(ra1Val - ra2[i + j]) + sinDecl1 * sinf(decl2[i + j]);
                degree = acos(cosAngle) * oneRad;
                if (!isnan(degree))
                {
                    // divide by 0.25 equals multiply by 4. xD
                    atomicAdd(&shared_histogram[(int)(degree * binsperdegree)], 1);
                    continue;
                }
                else
                {
                    atomicAdd(&shared_histogram[(cosAngle > 1.0f) ? 0 : (cosAngle < -1.0f) ? (totalBins - 1)
                                                                                           : (int)(degree * binsperdegree)],
                              1);
                }
            }
        }
    }
    __syncthreads(); // Ensure all threads have finished with shared memory

    id = threadIdx.x;
    while (id < totalBins)
    {
        atomicAdd(&histogram[id], shared_histogram[id]);
        id += threadsperblock;
    }
}

int main(int argc, char *argv[])
{
    // int noofblocks;
    int readdata(char *argv1, char *argv2);
    int getDevice(int deviceno);
    int writedata(char *argv3);
    void calculateOmega();
    unsigned long int histogramDRsum = 0L, histogramDDsum = 0L, histogramRRsum = 0L;
    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;
    cudaError_t myError;

    if (argc != 4)
    {
        printf("Usage: a.out real_data random_data output_data\n");
        return (-1);
    }

    if (getDevice(0) != 0)
        return (-1);

    if (readdata(argv[1], argv[2]) != 0)
        return (-1);
    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;
    // allocate mameory on the GPU
    cudaMalloc((void **)&ra_real_dev, NoofReal * sizeof(float));
    cudaMalloc((void **)&decl_real_dev, NoofReal * sizeof(float));
    cudaMalloc((void **)&ra_sim_dev, NoofSim * sizeof(float));
    cudaMalloc((void **)&decl_sim_dev, NoofSim * sizeof(float));

    cudaMalloc((void **)&histogramDR_dev, totalBins * sizeof(unsigned int));
    cudaMalloc((void **)&histogramDD_dev, totalBins * sizeof(unsigned int));
    cudaMalloc((void **)&histogramRR_dev, totalBins * sizeof(unsigned int));
    // copy data to the GPU

    cudaMemcpyAsync(ra_real_dev, ra_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(decl_real_dev, decl_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(ra_sim_dev, ra_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(decl_sim_dev, decl_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // calculate no of blocks and threads
    noofblocks = (NoofSim + threadsperblock - 1) / threadsperblock;
    printf("    # of blocks = %d ,    # threads in block = %d, # of total threades = %d \n", noofblocks, threadsperblock, threadsperblock * noofblocks);

    // run the kernels on the GPU
    FillHistogram<<<noofblocks, threadsperblock>>>(histogramDD_dev, NoofReal, ra_real_dev, decl_real_dev, NoofReal, ra_real_dev, decl_real_dev);
    FillHistogram<<<noofblocks, threadsperblock>>>(histogramRR_dev, NoofSim, ra_sim_dev, decl_sim_dev, NoofSim, ra_sim_dev, decl_sim_dev);
    FillHistogram<<<noofblocks, threadsperblock>>>(histogramDR_dev, NoofReal, ra_real_dev, decl_real_dev, NoofSim, ra_sim_dev, decl_sim_dev);
    // copy the results back to the CPU

    histogramDD = (unsigned int *)malloc(totalBins * sizeof(unsigned int));
    histogramRR = (unsigned int *)malloc(totalBins * sizeof(unsigned int));
    histogramDR = (unsigned int *)malloc(totalBins * sizeof(unsigned int));

    cudaMemcpyAsync(histogramDD, histogramDD_dev, totalBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(histogramRR, histogramRR_dev, totalBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(histogramDR, histogramDR_dev, totalBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free device memory
    cudaFree(ra_real_dev);
    cudaFree(decl_real_dev);
    cudaFree(ra_sim_dev);
    cudaFree(decl_sim_dev);

    cudaFree(histogramDD_dev);
    cudaFree(histogramRR_dev);
    cudaFree(histogramDR_dev);

    // calculate omega values on the CPU
    myError = cudaGetLastError();
    if (myError != cudaSuccess)
    {
        printf("    CUDA error: %s\n", cudaGetErrorString(myError));
        return (-1);
    }

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;
    kerneltime += end - start;
    printf("    Kernel time = %f seconds\n\n", kerneltime);

    for (int i = 0; i < totalBins; ++i)
    {
        histogramDDsum += histogramDD[i];
    }
    printf("    histogramDDsum = %ld\n", histogramDDsum);
    for (int i = 0; i < totalBins; ++i)
    {
        histogramRRsum += histogramRR[i];
    }
    printf("    histogramRRsum = %ld\n", histogramRRsum);
    for (int i = 0; i < totalBins; ++i)
    {
        histogramDRsum += histogramDR[i];
    }
    printf("    histogramDRsum = %ld\n", histogramDRsum);

    omega = (float *)malloc(totalBins * sizeof(float));

    calculateOmega();

    writedata(argv[3]);
    return (0);
}

void calculateOmega()
{
    for (int i = 0; i < totalBins; ++i)
    {
        if (histogramRR[i] != 0)
        {
            omega[i] = (float)(histogramDD[i] - 2 * histogramDR[i] + histogramRR[i]) / histogramRR[i];
        }
    }
}

int writedata(char *argv3)
{
    int i;
    FILE *outfil;

    outfil = fopen(argv3, "w");
    if (outfil == NULL)
    {
        printf("Cannot open output file %s\n", argv3);
        return (-1);
    }
    fprintf(outfil, "#bin,start/deg,omega,hist_DD,hist_DR,hist_RR\n");
    for (i = 0; i < totalBins; ++i)
    {
        fprintf(outfil, "%d,%.3f,%.6f,%u,%u,%u\n", i, (i * 0.25), omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
    }

    fclose(outfil);

    return (0);
}

int readdata(char *argv1, char *argv2)
{
    int i, linecount;
    char inbuf[180];
    double ra, dec, phi, theta, dpi;
    FILE *infil;

    printf("   Assuming input data is given in arc minutes!\n");
    // spherical coordinates phi and theta:
    // phi   = ra/60.0 * dpi/180.0;
    // theta = (90.0-dec/60.0)*dpi/180.0;

    dpi = acos(-1.0);
    infil = fopen(argv1, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv1);
        return (-1);
    }

    // read the number of galaxies in the input file
    int announcednumber;
    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv1);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 180, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv1, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv1, announcednumber, linecount);
        return (-1);
    }

    NoofReal = linecount;
    ra_real = (float *)calloc(NoofReal, sizeof(float));
    decl_real = (float *)calloc(NoofReal, sizeof(float));

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv1);
            fclose(infil);
            return (-1);
        }
        ra_real[i] = ra / 60.0 * (M_PI / 180.0);
        decl_real[i] = dec / 60.0 * (M_PI / 180.0);
        ++i;
    }

    fclose(infil);

    if (i != NoofReal)
    {
        printf("   Cannot read %s correctly\n", argv1);
        return (-1);
    }

    infil = fopen(argv2, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv2);
        return (-1);
    }

    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv2);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv2, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv2, announcednumber, linecount);
        return (-1);
    }

    NoofSim = linecount;
    ra_sim = (float *)calloc(NoofSim, sizeof(float));
    decl_sim = (float *)calloc(NoofSim, sizeof(float));

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv2);
            fclose(infil);
            return (-1);
        }
        ra_sim[i] = ra / 60.0 * (M_PI / 180.0);
        decl_sim[i] = dec / 60.0 * (M_PI / 180.0);
        ++i;
    }

    fclose(infil);

    if (i != NoofSim)
    {
        printf("   Cannot read %s correctly\n", argv2);
        return (-1);
    }

    return (0);
}

int getDevice(int deviceNo)
{

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128)
        return (-1);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name, device);
        printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
        printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                   =   %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels             =   ");
        if (deviceProp.concurrentKernels == 1)
            printf("     yes\n");
        else
            printf("    no\n");
        printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
        if (deviceProp.deviceOverlap == 1)
            printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if (device != 0)
        printf("   Unable to set device 0, using %d instead", device);
    else
        printf("   Using CUDA device %d\n\n", device);

    return (0);
}
