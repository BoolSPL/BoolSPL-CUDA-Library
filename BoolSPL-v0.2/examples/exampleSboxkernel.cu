////////////////////////////////////////////////////////////////////////////
//
// Copyright @2017 Dusan and Iliya.  All rights reserved.
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

//Declaration for global variables and dinamic array use in S-box functions
int sizeSbox, binary = 0;
int *SboxElemet;

bool CheckFile = true;

//Help Heder file - CPU computing boolean functions properties
#include "func_Boolean_CPU.h" 

//Heder file 2D DynamicArray
#include "2D_DynamicArray.h"

//Declaration for global variables use for S-box properties
int Lin_cpu, nl_cpu, Diff_cpu, AC_cpu, ADmax_cpu, ADmin_cpu, ACn_cpu, Lin_return, AD_return, AC_return;

//Help Heder file - CPU computing S-box functions properties and other functions
#include "helpSboxfunct.h"

//Help Heder file - CPU computing S-box functions properties
#include "funct_Sbox_CPU.h"


using namespace std;

////////// main function - S-box example using of GPU - CPU function for computing properties ////////////////
int main()
{
	printf("\nExample S-box BoolSPLG Library algorithms.\n");

	//BoolSPLG Properties Library functions
	//BoolSPLGCheckProperties();
	BoolSPLGMinimalRequires();

	printf("1. Compute Properties of input S-box\n");

	//string use for name of the S-box input file
	string Sbox = "sbox"; //String for name of the file with inverse permutation and cyclic matrix

	//Call function in "helpSboxfunct.h" input S-box (permutation file)
	SetParSbox(Sbox); //Permutation file have number of element plus 1

	if (CheckFile == false)
	{
		cout << "\nInital file is not set\n\n";
		return 0;
	}
	//Set paramether to compute
	cout << "Configuration for next parameter: \n";
	cout << "Size S-box: " << sizeSbox << "\n";
	cout << "Binary: " << binary-1 << "\n";

	SboxElemet = (int *)malloc(sizeof(int)* sizeSbox); //Allocate memory for permutation

	//============Call function in "helpSboxfunct.h" open permutation file	
	readFromFileMatPerm(Sbox, SboxElemet);

	//Print input S-box or not print
	int input;
	cout << "\n\nInput 1, or other (print S-box or not): \n";
	cout << " Other integer: Not print (S-box); \n";
	cout << " 1: Print (S-box); \n";
	cout << "Input:";
	cin >> input;

	cout << "\nPrint S-box;\n"; //Print
	if (input == 1)
	{
		//function for print header file "funct_Sbox_CPU.h"
		Print_Result(sizeSbox, SboxElemet, sizeSbox);
	}
	printf("\n");

	//Declaration and Allocate memory blocks 
	int **STT = AllocateDynamicArray<int>(binary, sizeSbox);
	int *binary_num = (int *)malloc(sizeof(int)* binary);

	SetSTT(SboxElemet, STT, binary_num);

	//Computing S-box properties ##HeaderSboxProperties.h##
	MainSboxProperties(STT, SboxElemet);

	/////////////////////////////////////////////////////////////////////////////
	//Start GPU program
	/////////////////////////////////////////////////////////////////////////////

	//set size array
	int sizeSboxArray = sizeof(int)* sizeSbox;

	//device vectors
	int *device_Sbox; // , *device_CF, *device_Vect_out;

	//check CUDA component status
	cudaError_t cudaStatus;

	//@@ Allocate GPU memory here

	//input S-box device vector
	cudaStatus = cudaMalloc((void **)&device_Sbox, sizeSboxArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	// Copy S-box input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Sbox, SboxElemet, sizeSboxArray, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Component function
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox*sizeSbox;

		int *device_CF;

		//CF S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		int *CPU_CF = (int *)malloc(sizeof(int)* sizeSbox*sizeSbox);
		int *host_CF = (int *)malloc(sizeof(int)* sizeSbox*sizeSbox);

		//@set GRID
		int sizethread = sizeSbox;
		int sizeblok = sizeSbox;

		//Compute Component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== all CF are save in one array CPU_STT ======================
			GenTTComponentFunc(i, SboxElemet, CPU_CF, sizeSbox);
		}

		//@Compute Component function GPU - BoolSPLG Library function
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(host_CF, device_CF, sizeArray, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(EXIT_FAILURE);
		}

		//@Check result
		cout << "\nCheck result Component functions:";
		check_rez(CPU_CF, host_CF, sizeSbox*sizeSbox);

		//@Free memory
		cudaFree(device_CF);

		free(CPU_CF);
		free(host_CF);
	}	
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox;

		int *device_CF;

		//CF S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		int *CPU_CF = (int *)malloc(sizeof(int)* sizeSbox);
		int *host_CF = (int *)malloc(sizeof(int)* sizeSbox);

		//@set GRID
		int sizethread = BLOCK_SIZE;
		int sizeblok = sizeSbox / BLOCK_SIZE;

		//@counter for check result
		int br = 0;

		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== One CF is save in array CPU_STT ===========================
			GenTTComponentFuncVec(i, SboxElemet, CPU_CF, sizeSbox);

			//@Compute Component function GPU - BoolSPLG Library function
			ComponentFnVec_kernel<<<sizeblok, sizethread>>>(device_Sbox, device_CF, i);

			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(host_CF, device_CF, sizeArray, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				exit(EXIT_FAILURE);
			}
			br+=check_rez_return(CPU_CF, host_CF, sizeSbox);		
	}
		//@Check result
		cout << "\nCheck result Component functions:";
		if (br==sizeSbox)
			cout << "\nCheck: True\n";
		else
			cout << "\nCheck: False\n";

		//@Free memory
		cudaFree(device_CF);

		free(CPU_CF);
		free(host_CF);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Test print S-box
	//cudaMemcpy(SboxElemet, device_Sbox, sizeSbox, cudaMemcpyDeviceToHost);
	//Print_Result(sizeSbox, SboxElemet, sizeSbox);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Linear Approximation Table (LAT) - Linearity (Lin(S))
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//@Declaration of device vectors
	int *device_CF, *device_LAT;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox*sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_LAT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_LAT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library LAT(S) function
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int Lin_gpu = 0;
	//Lin_gpu = WalshSpecTranSboxGPU(device_Sbox, device_CF, device_LAT, sizeSbox);
	Lin_gpu = WalshSpecTranSboxGPU_ButterflyMax(device_Sbox, device_CF, device_LAT, sizeSbox, true);

	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_LAT);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	//Check result
	cout << "\nCheck result LAT(S):";
	cout << "\nLin(S)_cpu:" << Lin_cpu << "\n";
	cout << "Lin(S)_gpu:" << Lin_gpu << "\n";

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree (deg(S))
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//@Declaration of ANF device vector
	int *device_ANF;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox*sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_ANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_ANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmax_gpu = 0;
	//ADmax_gpu = MobiusTranSboxADGPU(device_Sbox, device_CF, device_ANF, sizeSbox);
	ADmax_gpu = AlgebraicDegreeSboxGPU_ButterflyMax(device_Sbox, device_CF, device_ANF, sizeSbox);

	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_ANF);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree (deg(S)) Bitwise
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////


	int NumOfBits = sizeof(unsigned long long int) * 8 , NumInt;
	//@Declaration of ANF device vector
	unsigned long long int *device_NumIntVecCF, *device_NumIntVecANF, *host_NumIntVecCF;
	int *host_CF, *host_max_values, *device_Vec_max_values;

	if (sizeSbox<16384) //limitation cîme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{	
		cout << "\n\nNumOfBits:" << NumOfBits << "\n";
		NumInt = (sizeSbox*sizeSbox) / NumOfBits;
		cout << "NumOfInt:" << NumInt << "\n\n";

		host_CF = (int *)malloc(sizeof(int)* sizeSbox*sizeSbox);
		host_max_values = (int *)malloc(sizeof(int)* NumInt);
		host_NumIntVecCF = (unsigned long long int *)malloc(sizeof(unsigned long long int)* NumInt);

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int)* NumInt;
		
		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_NumIntVecCF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_NumIntVecANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_Vec_max_values, sizeof(int)* NumInt);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	else
	{
		cout << "\n\nNumOfBits:" << NumOfBits << "\n";
		NumInt = sizeSbox / NumOfBits;
		cout << "NumOfInt:" << NumInt << " (for every component function)\n\n";

		host_CF = (int *)malloc(sizeof(int)* sizeSbox);
		host_max_values = (int *)malloc(sizeof(int)* NumInt);
		host_NumIntVecCF = (unsigned long long int *)malloc(sizeof(unsigned long long int)* NumInt);

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int)* NumInt;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_NumIntVecCF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_NumIntVecANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_Vec_max_values, sizeof(int)* sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmax_bitwise_gpu = 0;
	//ADmax_gpu = MobiusTranSboxADGPU(device_Sbox, device_CF, device_ANF, sizeSbox);
	ADmax_bitwise_gpu = BitwiseAlgebraicDegreeSboxGPU_ButterflyMax(SboxElemet, host_CF, host_max_values, host_NumIntVecCF, device_NumIntVecCF, device_NumIntVecANF, device_Vec_max_values, sizeSbox);
	//	BitwiseAlgebraicDegreeSboxGPU_ButterflyMax(SboxElemet, device_CF, device_ANF, sizeSbox);

	//@Free memory
	cudaFree(device_NumIntVecCF);
	cudaFree(device_NumIntVecANF);
	cudaFree(device_Vec_max_values);

	free(host_CF);
	free(host_max_values);
	free(host_NumIntVecCF);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	//Check result
	cout << "\nCheck result ANF(S):";
	cout << "\ndeg(S)_cpu:" << ADmax_cpu << "\n";
	cout << "deg(S)_gpu:" << ADmax_gpu << "\n";
	cout << "deg(S)_bitwise_gpu:" << ADmax_bitwise_gpu << "\n";

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Autocorrelation Transform (ACT) - Autocorrelation (AC)
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//@Declaration of device vectors
	int *device_ACT;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox*sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_ACT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void **)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void **)&device_ACT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ACT(S) function
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int AC_gpu = 0;
	//AC_gpu = AutocorrelationTranSboxGPU(device_Sbox, device_CF, device_ACT, sizeSbox);
	AC_gpu = AutocorrelationTranSboxGPU_ButterflyMax(device_Sbox, device_CF, device_ACT, sizeSbox, true);

	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_ACT);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	//Check result
	cout << "\nCheck result ACT(S):";
	cout << "\nAC(S)_cpu:" << ACn_cpu << "\n";
	cout << "AC(S)_gpu:" << AC_gpu << "\n";

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Difference Distribution Table (DDT) - Differential uniformity (DU)
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//@Declaration device DDT vector
	int *device_DDT;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox*sizeSbox;
		
		//DDT S-box device vector	
		cudaStatus = cudaMalloc((void **)&device_DDT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int)* sizeSbox;

		//DDT S-box device vector	
		cudaStatus = cudaMalloc((void **)&device_DDT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library DDT(S) function
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int Diff_gpu = 0;
	//Diff_gpu = DDTSboxGPU(device_Sbox, device_DDT, sizeSbox);
	Diff_gpu = DDTSboxGPU_ButterflyMax(device_Sbox, device_DDT, sizeSbox, true);
	//@Free memory
	cudaFree(device_DDT);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	//Check result
	cout << "\nCheck result DDT(S):";
	cout << "\nDiff(f)_cpu:" << Diff_cpu << "\n";
	cout << "Diff(f)_gpu:" << Diff_gpu << "\n";

	//free memory
	cudaFree(device_Sbox);

	FreeDynamicArray<int>(STT);
	free(binary_num);
	//************************

	free(SboxElemet);

	return 0;
}