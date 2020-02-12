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

//Main Library header file

//System includes
#include <stdio.h>
#include <iostream>
#include <string>

//CUDA runtime
#include "cuda_runtime.h"

#define BLOCK_SIZE 1024

//@ Global variable for grid
int sizeblok, sizeblok1, sizethread, sizefor, sizefor1;
int numThreadsRDM, numBlocksRDM, whichKernelRDM; //data for reduction Max operation

//BoolSPLG CUDA Properties header file
#include "CUDA_Properties.h"

//BoolSPLG base CUDA functions
#include "boolsplg_base_funct.cuh"

//BoolSPLG Boolean device functions
#include "boolsplg_dev_funct.cuh"

//BoolSPLG GPU reduction function heder file
#include "reduction.h"

//BoolSPLG Boolean Algorithms
#include "boolsplg_algorithms.cuh"

using namespace std;

/////////////////////////////////////////////////////////////////////////
//Declaration of function
/////////////////////////////////////////////////////////////////////////

////Declaration for CUDA Properties function
void printDevProp(cudaDeviceProp devProp); //0.1
void BoolSPLGCheckProperties(); //0.1
void BoolSPLGMinimalRequires(); //0.1
void BoolSPLGCheckProperties_v1(); //v0.2
void bandwidthTest(const unsigned int bytes, int nElements); //v0.2

//Declaration for CPU help function
int reduceCPU_max_libhelp(int *vals, int nvals); //0.1
int reduceCPU_min_libhelp(int *vals, int nvals); //0.1

//Declaration for base CUDA functions
//void cudaMallocBoolSPLG(int **d_vec, int sizeBoolean);
//void cudaMemcpyBoolSPLG_HtoD(int *d_vec, int *h_vec, int sizeBoolean);

//Function: Set GRID
inline void setgrid(int size); //0.1
inline void setgridBitwise(int size); //0.2

//Function: Return Most significant bit start from 0
unsigned int msb32(unsigned int x); //0.1

//Function: Check array size
inline void CheckSize(int size); //0.1
inline void CheckSizeSbox(int size); //v0.2
inline void CheckSizeBoolBitwise(int size); //v0.2
inline void CheckSizeSboxBitwise(int size); //v0.2
inline void CheckSizeSboxBitwiseMobius(int size); //v0.2

//Function: CPU set TT in 64 bit int variables and vise versa
void BinVecToDec(int size, int *Bin_Vec, unsigned long long int *NumIntVec, int NumInt); //v0.2
void DecVecToBin(int NumOfBits, int *Bin_Vec, unsigned long long int *NumIntVec, int NumInt); //v0.2
 
//Declaration for Boolean GPU device functions

//GPU Fast Walsh Transform
extern __global__ void fwt_kernel_shfl_xor_SM(int *VectorValue, int *VectorValueRez, int step); //0.1
extern __global__ void fwt_kernel_shfl_xor_SM_MP(int *VectorValue, int fsize, int fsize1); //0.1

//GPU Fast Mobius Transform
extern __global__ void fmt_kernel_shfl_xor_SM(int * VectorValue, int * VectorRez, int sizefor); //0.1
extern __global__ void fmt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1); //0.1

//GPU Bitwise Fast Mobius Transform
extern __global__ void fmt_bitwise_kernel_shfl_xor_SM(unsigned long long int *vect, unsigned long long int *vect_out, int sizefor, int sizefor1); //v0.2
extern __global__ void fmt_bitwise_kernel_shfl_xor_SM_MP(unsigned long long int * VectorValue, int fsize, int fsize1); //v0.2

//GPU compute Algebraic Degree
extern __global__ void kernel_AD(int *Vec); //0.1
extern __global__ void kernel_bitwise_AD(unsigned long long int *NumIntVec, int *Vec_max_values, int NumOfBits); //v0.2

//GPU Inverse Fast Walsh Transform
extern __global__ void ifmt_kernel_shfl_xor_SM(int * VectorValue, int * VectorValueRez, int step); //0.1
extern __global__ void ifmt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1); //0.1

//GPU Min-Max Butterfly
extern __global__ void Butterfly_max_min_kernel_shfl_xor_SM(int *VectorValue, int *VectorValueRez, int step); //v0.2
extern __global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1); //v0.2

extern __global__ void ifmt_kernel_shfl_xor_SM_Sbox(int * VectorValue, int * VectorValueRez, int step); //0.1

//Declaration for S-box GPU device functions

//GPU Bitwise Fast Mobius Transform
extern __global__ void fmt_bitwise_kernel_shfl_xor_SM_Sbox(unsigned long long int *vect, unsigned long long int *vect_out, int sizefor, int sizefor1); //v0.2

//GPU compute Algebraic Degree
extern __global__ void kernel_AD_Sbox(int *Vec); //0.1
extern __global__ void kernel_bitwise_AD_Sbox(unsigned long long int *NumIntVecANF, int *max_values, int NumOfBits); //v0.2

//GPU Difference Distribution Table
extern __global__ void DDTFnAll_kernel(int *Sbox_in, int *DDT_out, int n); //0.1
extern __global__ void DDTFnVec_kernel(int *Sbox_in, int *DDT_out, int row); //0.1

//GPU S-box Component functions 
extern __global__ void ComponentFnAll_kernel(int *Sbox_in, int *CF_out, int n); //0.1
extern __global__ void ComponentFnVec_kernel(int *Sbox_in, int *CF_out, int row); //0.1

//Declaration for Boolean procedures
int WalshSpecTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction); //0.1 BoolFWT_compute
void MobiusTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size);		//0.1 BoolFMT_compute
int AlgebraicDegreeBoolGPU(int *device_Vect, int *device_Vect_rez, int size);	//0.1 BoolAD_compute
int AutocorrelationTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction);	//0.1BoolAC_compute

int WalshSpecTranBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction); //v0.2 BoolFWT_compute
int AlgebraicDegreeBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size);							 //v0.2 BoolAD_compute
int AutocorrelationTranBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction);  //v0.2 BoolAC_compute

//Declaration for Bitwise Boolean procedures 
void BitwiseMobiusTranBoolGPU(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int size); //v0.2 Bitwise BoolFMT_compute
int BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int *device_Vec_max_values, int *host_Vec_max_values, int size); //v0.2

//Declaration for S-box procedures 
int WalshSpecTranSboxGPU(int *device_Sbox, int *device_CF, int *device_LAT, int sizeSbox); //0.1
int MobiusTranSboxADGPU(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox); //0.1
int AutocorrelationTranSboxGPU(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox); //0.1

int WalshSpecTranSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_LAT, int sizeSbox, bool returnMax); //v0.2
void MobiusTranSboxGPU(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox); //v0.2
int AlgebraicDegreeSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox); //v0.2
int AutocorrelationTranSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox, bool returnMax); //v0.2

int DDTSboxGPU(int *device_Sbox, int *device_DDT, int sizeSbox); //0.1
int DDTSboxGPU_ButterflyMax(int *device_Sbox, int *device_DDT, int sizeSbox, bool returnMax); //v0.2

//Declaration for Bitwise S-box procedures 
void BitwiseMobiusTranSboxGPU(int *host_Sbox, int *host_Vect_CF, unsigned long long int *host_NumIntVecCF, unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int sizeSbox); //v0.2
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMax(int *host_Sbox, int *host_Vect_CF, int *host_max_values, unsigned long long int *host_NumIntVecCF, unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int *device_Vec_max_values, int sizeSbox); //v0.2

//Declaration for Max - Min Reduction function
int runReductionMax(int size, int *d_idata); //0.1
int runReductionMin(int size, int *d_idata); //v0.2

//Declaration for Butterfly max function
int Butterfly_max_kernel(int sizeSbox, int *device_data); //v0.2

