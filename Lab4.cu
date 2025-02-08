/*
Author: Bantwal Vaibhav Mallya
Class: ECE 6122 A
Last Date Modified: 9th November 2023

Description:
This file is a CUDA program to simulate 2D random walk 

*/


#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include<chrono>
using namespace std;
__global__  void cudaRandomWalk(int seed, float *xCoord, float *yCoord, int *direction, int steps,int walkers, float *distance, float *avgDist){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	curandState cudaState;
        curand_init(seed,tid,0,&cudaState); //initializing the random number generator function
	if(tid < walkers){
	  for(int i=0; i < steps; i=i+1){
	       *direction = (int)(curand_uniform(&cudaState)*4); 
	       if(*direction == 0){
	         yCoord[tid] = yCoord[tid]+1; //North
	       }
	       else if(*direction == 1){
	         yCoord[tid] = yCoord[tid]-1; //South
	       }

	       else if(*direction == 2){
	       	 xCoord[tid] = xCoord[tid]-1; //West 
	       }
	       else{
	         xCoord[tid] = xCoord[tid]+1; //East
	       }
	  }
	  
	  distance[tid] = sqrt(xCoord[tid]*xCoord[tid] + yCoord[tid]*yCoord[tid])/walkers;

	  //atomic add done to prevent race conditions
	  atomicAdd(avgDist,distance[tid]); 
	}
	
}

void onlyCudaMalloc( int stepsCuda, int walkersCuda){

        auto startTimeCM = std::chrono::high_resolution_clock::now();
        float *xHost,*yHost, *avgDistHost;

	//creating vectors and allocating memory for the vectors in the host
	xHost = (float*)malloc(sizeof(float)*walkersCuda);
	yHost = (float*)malloc(sizeof(float)*walkersCuda);
	avgDistHost = (float*)malloc(sizeof(float));

	//initializing the x and y coordinates of walkers to zero (origin)
	for(int i=0;i<walkersCuda;i++){
	  xHost[i] = 0.0f;
	  yHost[i] = 0.0f;
	}

        
        //creating variables for the device
	//distanceCuda gives the distance of each walker
	//avgDistDev gives the average distance of all walkers from the origin 
	//direction gives the random direction in each step 
	float *xCuda,*yCuda,*distanceCuda, *avgDistDev;
	int  *directionCuda;
	int k = sizeof(float)*walkersCuda;


	//allocating memory in device
	cudaMalloc((void**) &xCuda,k); 
	cudaMalloc((void**) &yCuda,k); 
	cudaMalloc((void**) &directionCuda, sizeof(int));
	cudaMalloc((void**) &distanceCuda,k); 
	cudaMalloc((void**) &avgDistDev, sizeof(float));

	//transfer data from host to device
	cudaMemcpy(xCuda, xHost, k, cudaMemcpyHostToDevice);
	cudaMemcpy(yCuda, yHost, k, cudaMemcpyHostToDevice);

	int blockSize = 256; //threads per block 
	int gridSize = ((walkersCuda + blockSize)/blockSize); //total blocks

	int seed = 1234;
	
	//executing the random walk kernel
	cudaRandomWalk<<<gridSize,blockSize>>>(seed,xCuda, yCuda, directionCuda, stepsCuda, walkersCuda, distanceCuda, avgDistDev);

	cudaDeviceSynchronize();

	//transfer data from device to host
	cudaMemcpy(avgDistHost, avgDistDev, sizeof(float), cudaMemcpyDeviceToHost);

        auto endTimeCM = std::chrono::high_resolution_clock::now();
	auto durationCM = std::chrono::duration_cast<std::chrono::microseconds>(endTimeCM - startTimeCM);

	printf("\n Normal CUDA Memory Allocation:");
	printf("\n     Average distance from origin = %f", *avgDistHost);
	printf("\n     Time to calculate(microsec): %ld",static_cast<long>(durationCM.count()));

	//deallocate memory in device
	cudaFree(xCuda);
	cudaFree(yCuda);
	cudaFree(distanceCuda);
	cudaFree(avgDistDev);

	//deallocate memory in host
	free(xHost);
	free(yHost);
	free(avgDistHost);
}

void onlyCudaMallocHost(int stepsCuda, int walkersCuda){
        

     	auto startTimeCMH = std::chrono::high_resolution_clock::now();

     	float *xHost, *yHost,  *avgDistHost; 
	
	int k = sizeof(float)*walkersCuda;

	
	cudaMallocHost((void**) &xHost,k);
	cudaMallocHost((void**) &yHost,k);
	cudaMallocHost((void**) &avgDistHost,sizeof(float));
     	

        //creating variables for the device
	//distanceCuda gives the distance of each walker
	//avgDistDev gives the average distance of all walkers from the origin 
	//direction gives the random direction in each step 
	float *xCuda,*yCuda,*distanceCuda, *avgDistDev;
	int  *directionCuda;
	

	//initializing the x and y coordinates of walkers to zero (origin)
	for(int i=0;i<walkersCuda;i++){
	  xHost[i] = 0.0f;
	  yHost[i] = 0.0f;
	}

	//allocating memory in device
	
	cudaMalloc((void**) &xCuda,k); 
	cudaMalloc((void**) &yCuda,k);
	cudaMalloc((void**) &directionCuda, sizeof(int));
	cudaMalloc((void**) &distanceCuda, k);
	cudaMalloc((void**) &avgDistDev, sizeof(float));

	//transfer data from host to device
	cudaMemcpy(xCuda, xHost, k, cudaMemcpyHostToDevice);
	cudaMemcpy(yCuda, yHost, k, cudaMemcpyHostToDevice);

	int blockSize = 256; //threads per block
	int gridSize = ((walkersCuda + blockSize)/blockSize); //total blocks

	int seed = 1234;
	
	//executing random walk kernel
	cudaRandomWalk<<<gridSize,blockSize>>>(seed,xCuda, yCuda, directionCuda, stepsCuda, walkersCuda, distanceCuda, avgDistDev);

	//synchronizing the device with the host
	cudaDeviceSynchronize();

	//transfer data from device to host
	cudaMemcpy(avgDistHost, avgDistDev, sizeof(float), cudaMemcpyDeviceToHost);

	auto endTimeCMH = std::chrono::high_resolution_clock::now();
	auto durationCMH = std::chrono::duration_cast<std::chrono::microseconds>(endTimeCMH - startTimeCMH);

	printf("\n Pinned CUDA Memory Allocation:");
	printf("\n     Average distance from origin = %f", *avgDistHost);
	printf("\n     Time to calculate(microsec): %ld",static_cast<long>(durationCMH.count()));

	//deallocate memory in device
	cudaFree(xCuda);
	cudaFree(yCuda);
	cudaFree(distanceCuda);
	cudaFree(avgDistDev);

	//deallocate memory in host	
	cudaFree(xHost);
	cudaFree(yHost);
	cudaFree(avgDistHost);
}

void onlyCudaMallocManaged( int stepsCuda, int walkersCuda){

        auto startTimeCMM = std::chrono::high_resolution_clock::now();

        float *xHost,*yHost,*avgDistHost;
	int k = sizeof(float)*walkersCuda;

	xHost = (float*)malloc(k);
	yHost = (float*)malloc(k);
	avgDistHost = (float*)malloc(sizeof(float));

	//initializing the x and y coordinates of walkers to zero (origin)
	for(int i=0;i<walkersCuda;i++){
	  xHost[i] = 0.0f;
	  yHost[i] = 0.0f;
	}
        

        //creating variables for the device
	//distanceCuda gives the distance of each walker
	//avgDistDev gives the average distance of all walkers from the origin 
	//direction gives the random direction in each step
	float *xCuda,*yCuda,*distanceCuda, *avgDistDev;
	int  *directionCuda; 

	//allocating memory in device
	cudaMallocManaged((void**) &xCuda, k); 
	cudaMallocManaged((void**) &yCuda, k); 
	cudaMallocManaged((void**) &directionCuda, sizeof(int));
	cudaMallocManaged((void**) &distanceCuda, k); 
	cudaMallocManaged((void**) &avgDistDev, sizeof(float));

	//transfer data from host to device
	cudaMemcpy(xCuda, xHost, k, cudaMemcpyHostToDevice);
	cudaMemcpy(yCuda, yHost, k, cudaMemcpyHostToDevice);

	int blockSize = 256; //threads per block
	int gridSize = ((walkersCuda + blockSize)/blockSize); //total blocks
	int seed = 1234;
	
	//executing random walk kernel
	cudaRandomWalk<<<gridSize,blockSize>>>(seed,xCuda, yCuda, directionCuda, stepsCuda, walkersCuda, distanceCuda, avgDistDev);

	//synchronizing the device with the host 
	cudaDeviceSynchronize();

	//transfer data from device to host
	cudaMemcpy(avgDistHost, avgDistDev, sizeof(float), cudaMemcpyDeviceToHost);

	auto endTimeCMM = std::chrono::high_resolution_clock::now();
	auto durationCMM = std::chrono::duration_cast<std::chrono::microseconds>(endTimeCMM - startTimeCMM);

	printf("\n Managed CUDA Memory Allocation:");
	printf("\n     Average distance from origin = %f", *avgDistHost);
	printf("\n     Time to calculate(microsec): %ld",static_cast<long>(durationCMM.count()));
	printf("\n Bye");

	//deallocate memory in device
	cudaFree(xCuda);
	cudaFree(yCuda);
	cudaFree(distanceCuda);
	cudaFree(avgDistDev);

	//deallocate memory in host 
	free(xHost);
	free(yHost);
	free(avgDistHost);
}

int main(int argc, char *argv[]){

	//W = number of walkers, I = number of steps
	int W=0, I=0;  
	int inputFlag;

	//code for input flags for walkers (-W) and steps (-I)
	while((inputFlag = getopt(argc,argv,"W:I:")) != -1){
	  switch(inputFlag){
	    case 'W':
		W = atoi(optarg);
		break;
	    case 'I':
	        I = atoi(optarg);
		break;
	    default:
	        fprintf(stderr, "Usage: %s -W <value> -I <value>\n",argv[0]);
		exit(EXIT_FAILURE);
	    
	  }
	}

	//calling function that uses cudaMalloc 
	onlyCudaMalloc(I, W);
	
	//calling function that uses cudaMallocHost
	onlyCudaMallocHost(I, W);

	//calling function that uses cudaMallocManaged 	
	onlyCudaMallocManaged(I,W);
	

}

