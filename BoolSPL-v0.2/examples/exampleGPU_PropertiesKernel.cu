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

using namespace std;

////////// main function - functions which show GPU - CUDA Capable device(s) characteristics (Utilities) ////////////////
int main()
{
	cout << "\n=============================================================================";
	printf("\nExample, functions which show GPU - CUDA Capable device(s) characteristics.\n");
	
	//BoolSPLG Properties Library functions

	cout << "=============================================================================\n";
	//Function to check if it fulfill BoolSPLG CUDA-capable requires 
	BoolSPLGMinimalRequires();
	
	cout << "\n=============================================================================\n";
	//Function Detected and show CUDA Capable device(s) characteristics
	BoolSPLGCheckProperties();
	
	cout << "\n=============================================================================\n";
	//Function Detected and show CUDA Capable device(s) characteristics (full, extend informacion)
	BoolSPLGCheckProperties_v1();
	
	cout << "\n=============================================================================\n";
	//Simple test function to measure the memcopy bandwidth of the GPU
	int size = 1;
	cout << "Input memory size for transfer (MB):";
	cin >> size;
	unsigned int nElements = size*256*1024;
	const unsigned int bytes = nElements * sizeof(int);
	bandwidthTest(bytes, nElements);
	
	cout << "=============================================================================\n";

	return 0;
}