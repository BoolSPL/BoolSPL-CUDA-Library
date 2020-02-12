////////////////////////////////////////////////////////////////////////////
//
// Copyright @2017-2019 Dusan and Iliya.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <iostream>

// CUDA runtime.
#include "cuda_runtime.h"

//Main Library header file
#include "BoolSPLG_v02.cuh" 

//Help Heder file - CPU computing boolean functions properties
#include "func_Boolean_CPU.h"

using namespace std;

////////// main function - Boolean example using of GPU - CPU function for computing properties ////////////////
int main()
{
	cout << "==========================================================";
	printf("\nExample Boolean function BoolSPLG Library algorithms.\n");
	cout << "==========================================================";
	//BoolSPLG Properties Library functions
	BoolSPLGMinimalRequires();
	cout << "==========================================================\n";
	//set size of Boolean vector
	int size = 1048576;
	printf("Input size:%d\n", size);

	//Declaration, host_Vect_CPU - vector for CPU computation, host_Vect_GPU -vector for GPU computation, walshvec - Store result from CPU computation
	int  *host_Vect_TT, *host_Vect_PTT, *host_Vect_rez, *walshvec_cpu, *anf_cpu, *rf_cpu;
	int Lin_cpu, AC_cpu, degMax_cpu;

	//Bitwise ANF computation
	unsigned long long int *host_NumIntVecTT, *host_NumIntVecANF; // , *mack_vec_Int;

	//Allocate memory block. Allocates a block of size bytes of memory
	host_Vect_TT = (int *)malloc(sizeof(int)* size);
	host_Vect_PTT = (int *)malloc(sizeof(int)* size);
	host_Vect_rez = (int *)malloc(sizeof(int)* size);
	walshvec_cpu = (int *)malloc(sizeof(int)* size);
	anf_cpu = (int *)malloc(sizeof(int)* size);
	rf_cpu = (int *)malloc(sizeof(int)* size);

	//bitwise variable and memory
	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = size / NumOfBits;

	//Allocate memory block for bitwise computation
	host_NumIntVecTT = (unsigned long long int *)malloc(sizeof(unsigned long long int)* NumInt);
	host_NumIntVecANF = (unsigned long long int *)malloc(sizeof(unsigned long long int)* NumInt);

	int *host_max_values_AD = (int *)malloc(sizeof(int)* size);

	//Function: Fill vector for FWT computation
	Fill_dp_vector(size, host_Vect_TT, host_Vect_PTT);
	cout << "\nPrint input vector:\n";
	//	Print_Result(size, host_Vect_TT, size);

	//Help Heder file "func_Boolean_CPU.h" contein CPU Boolean properties function
	//Function: Fast Walsh Transformation function CPU (W_f(f))
	FastWalshTrans(size, host_Vect_PTT, walshvec_cpu);
	Lin_cpu = reduceCPU_max(walshvec_cpu, size);
	//Function: Fast Mobius Transformation function CPU (ANF(f))
	FastMobiushTrans(size, host_Vect_TT, anf_cpu);
	degMax_cpu = FindMaxDeg(size, anf_cpu);
	//Function: Compute Autocorrelation CPU (r_f(f))
	FastWalshTrans(size, host_Vect_PTT, rf_cpu);
	PTT_fun_pow2(size, rf_cpu);
	FastWalshTransInv(size, rf_cpu);
	AC_cpu = reduceCPU_AC(size, rf_cpu);

	//=== Bitwise comptation ====

	//convert TT bool to integers
	BinVecToDec(NumOfBits, host_Vect_TT, host_NumIntVecTT, NumInt);

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// start CPU bitwise function
	CPU_FWT_bitwise(host_NumIntVecTT, host_NumIntVecANF, NumOfBits, NumInt);
	//conversion from decimal to binary and fine deg(f)
	int degMax_bitwise_cpu = DecVecToBin_maxDeg(NumOfBits, host_NumIntVecANF, NumInt);
	/////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////
	//Start GPU program
	/////////////////////////////////////////////////////////////////////////////

	//set size array
	int sizeBoolean = sizeof(int)* size;

	//device vectors
	int *device_Vect, *device_Vect_rez;

	unsigned long long int *device_NumIntVecTT, *device_NumIntVecANF;
	int *device_max_values_AD;

	/////////////////////////////////////////////////////////////////////////////
	//set Fast Walsh Transform
	/////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory here

	//check CUDA component status
	cudaError_t cudaStatus;

	//input device vector
	cudaStatus = cudaMalloc((void **)&device_Vect, sizeBoolean);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	//output device vector
	cudaStatus = cudaMalloc((void **)&device_Vect_rez, sizeBoolean);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//input integer TT device vector
	cudaStatus = cudaMalloc((void **)&device_NumIntVecTT, sizeof(unsigned long long int)* NumInt);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//output integer ANF device vector
	cudaStatus = cudaMalloc((void **)&device_NumIntVecANF, sizeof(unsigned long long int)* NumInt);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	
	//device max AD on every integer
	cudaStatus = cudaMalloc((void **)&device_max_values_AD, sizeof(int)* NumInt);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Vect, host_Vect_PTT, sizeBoolean, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FWT(f) function - two diffrent functions for FWT(f) calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int Lin_gpu = 0;
	//Lin_gpu = WalshSpecTranBoolGPU(device_Vect, device_Vect_rez, size, true); //use reduction Max
	Lin_gpu = WalshSpecTranBoolGPU_ButterflyMax(device_Vect, device_Vect_rez, size, true); //use Butterfly Max
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_Vect_rez, device_Vect_rez, sizeBoolean, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//Print CPU FWT result
	cout << "\nPrint FWT(f) CPU:\n";
	//Print_Result(size, walshvec_cpu, size);

	//Check result
	cout << "\nCheck result FWT(f):";
	//Help Heder file "func_Boolean_CPU.h" function
	check_rez(host_Vect_rez, walshvec_cpu, size);

	cout << "\nLin(f)_cpu:" << Lin_cpu << "\n";
	cout << "Lin(f)_gpu:" << Lin_gpu << "\n";

	//Print GPU FWT result
	cout << "\nPrint FWT(f) GPU:\n";
	//Print_Result(size, host_Vect_rez, size);

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Set Fast Mobius Transform
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Vect, host_Vect_TT, sizeBoolean, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FMT(f) function
	//////////////////////////////////////////////////////////////////////////////////////////////////
	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_Vect_rez, device_Vect_rez, sizeBoolean, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//Check FMT(f) result
	cout << "\nCheck result FMT(f):";
	check_rez(host_Vect_rez, anf_cpu, size);

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FMT(f) function - two diffrent functions for FMT(f) - Algebraic Degree calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int degMax_gpu=0;
	//degMax_gpu = AlgebraicDegreeBoolGPU(device_Vect, device_Vect_rez, size); //use reduction Max
	degMax_gpu = AlgebraicDegreeBoolGPU_ButterflyMax(device_Vect, device_Vect_rez, size); //use Butterfly Max
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_Vect_rez, device_Vect_rez, sizeBoolean, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Set Bitwise Fast Mobius Transform
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//convert bool into integers
	BinVecToDec(NumOfBits, host_Vect_TT, host_NumIntVecTT, NumInt);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_NumIntVecTT, host_NumIntVecTT, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy input integer TT vectors from host memory to GPU buffer
	//cudaMemcpy(device_NumIntVecTT, host_NumIntVecTT, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library bitwise FMT(f) function
	//////////////////////////////////////////////////////////////////////////////////////////////////
	BitwiseMobiusTranBoolGPU(device_NumIntVecTT, device_NumIntVecANF, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_NumIntVecANF, device_NumIntVecANF, sizeof(unsigned long long int)* NumInt, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//conversion from decimal to binary on ANF array
	DecVecToBin(NumOfBits, host_Vect_rez, host_NumIntVecANF, NumInt);

	//Check FMT(f) result
	cout << "\nCheck result bitwise FMT(f):";
	check_rez(host_Vect_rez, anf_cpu, size);

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FMT(f) function - Bitwise Algebraic Degree calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int degMax_bitwise_gpu = 0;
	degMax_bitwise_gpu = BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(device_NumIntVecTT, device_NumIntVecANF, device_max_values_AD, host_max_values_AD, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////

	// Print deg(f) result
	cout << "\nCheck result deg(f):";
	cout << "\ndeg(f)max_cpu:" << degMax_cpu << "\n";
	cout << "deg(f)max_cpu(bitwise):" << degMax_bitwise_cpu << "\n";
	cout << "deg(f)max_gpu:" << degMax_gpu << "\n";
	cout << "deg(f)max_gpu(bitwise):" << degMax_bitwise_gpu << "\n";
	

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Set Function to compute Autocorrelation (r_f(f))
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Vect, host_Vect_PTT, sizeBoolean, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ACT(f) function - two diffrent functions for ACT(f) calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int AC_gpu = 0;
	//AC_gpu = AutocorrelationTranBoolGPU(device_Vect, device_Vect_rez, size, true); //use reduction Max
	AC_gpu = AutocorrelationTranBoolGPU_ButterflyMax(device_Vect, device_Vect_rez, size, true); //use Butterfly Max
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_Vect_rez, device_Vect_rez, sizeBoolean, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//Check result
	cout << "\nCheck result AC(f):";
	check_rez(host_Vect_rez, rf_cpu, size);
	cout << "\nAC(f)_cpu:" << AC_cpu << "\n";
	cout << "AC(f)_gpu:" << AC_gpu << "\n";

	//@Free memory
	cudaFree(device_Vect);
	cudaFree(device_Vect_rez);

	free(host_Vect_TT);
	free(host_Vect_PTT);
	free(host_Vect_rez);
	free(walshvec_cpu);
	free(anf_cpu);
	free(rf_cpu);

	free(host_NumIntVecTT);
	free(host_NumIntVecANF);

	free(host_max_values_AD);

	return 0;
}