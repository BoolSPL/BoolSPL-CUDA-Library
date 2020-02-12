//BoolSPLG Boolean device functions
// System includes
#include <stdio.h>
#include <iostream>
//
//#include "BoolSPLG_api.h"
//
//#define BLOCK_SIZE 1024
//
//using namespace std;

//*************************************************************************************************************
//Functions: Boolean
//*************************************************************************************************************

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fwt_kernel_shfl_xor_SM(int *VectorValue, int *VectorValueRez, int step)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID 
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
	}
	//"value" now contains the sum across all threads 

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function: Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fwt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID 
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
	}
	// "value" now contains the sum across all threads 

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Invers Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ifmt_kernel_shfl_xor_SM(int * VectorValue, int * VectorValueRez, int step)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID 
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
	}
	//"value" now contains the sum across all threads 

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function: Invers Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ifmt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID 
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
	}
	// "value" now contains the sum across all threads 

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM(int *VectorValue, int *VectorValueRez, int step)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID 
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
	}
	//"value" now contains the sum across all threads 

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}
	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function: Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID 
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
	}
	// "value" now contains the sum across all threads 

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			//value = (value + tmpsdata[tid + j]);
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			//value = (-value + tmpsdata[tid - j]);
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}
	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


//*************************************************************************************************************
//Function forS-box
//*************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Invers Fast Walsh Transforms for S-box => sizeSbox <= BLOCK_SIZE
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ifmt_kernel_shfl_xor_SM_Sbox(int * VectorValue, int * VectorValueRez, int step)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID 
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0, ZeroOne = 0;
	//Use XOR mode to perform butterfly reduction 
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
	}
	//"value" now contains the sum across all threads 

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//(i - (i / n)*n) = > i%n
	//ZeroOne = -((tid - 1) - ((tid - 1) / (tid + 1))*(tid + 1)) + tid; //=>ZeroOne = -((tid - 1) % (tid + 1)) + tid;

	ZeroOne = tid && 1; //=> 0 1 1 1 1 ...

	//save value in global memory
	VectorValueRez[laneId] = value*ZeroOne;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_kernel_shfl_xor_SM(int * VectorValue, int * VectorRez, int sizefor)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	// Seed starting value as inverse lane ID 
	int value = VectorValue[i];
	int f1, r = 1;

	for (int j = 1; j<32; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//printf("j %d, r %d, f1 %d, val: %d, __shfl_:%d\n", j, r, f1, value,__shfl_down(value, j));
		value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor; j *= 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata[tid - j];
		}
		__syncthreads();
	}
	//save value in global memory
	VectorRez[i] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function: Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (i - (i / fsize)*fsize) * 1024 + (i / fsize); //laneId%n

	// Seed starting value as inverse lane ID 
	int value = VectorValue[ji];
	int f1, r = 1;

	for (int j = 1; j<fsize1; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//printf("j %d, r %d, f1 %d, val: %d, __shfl_:%d\n", j, r, f1, value,__shfl_down(value, j));
		value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<fsize; j *= 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata[tid - j];
		}
		__syncthreads();
	}
	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Bitwise Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_bitwise_kernel_shfl_xor_SM(unsigned long long int *vect, unsigned long long int *vect_out, int sizefor, int sizefor1)
{
	//declaration for shared memory 
	extern __shared__ unsigned long long int tmpsdata1[];

	unsigned int tid = threadIdx.x;
	unsigned int ij = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	unsigned long long int value = vect[ij];

	value ^= (value & 12297829382473034410) >> 1;
	value ^= (value & 14757395258967641292) >> 2;
	value ^= (value & 17361641481138401520) >> 4;
	value ^= (value & 18374966859414961920) >> 8;
	value ^= (value & 18446462603027742720) >> 16;
	value ^= (value & 18446744069414584320) >> 32;

	int f1, r = 1;

	for (int j = 1; j<sizefor; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//printf("j %d, r %d, f1 %d, val: %d, __shfl_:%d\n", j, r, f1, value,__shfl_down(value, j));
		value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor1; j *= 2)
	{
		tmpsdata1[tid] = value;
		__syncthreads();

		if ((tid&j) == j)
		{
			value = value^tmpsdata1[tid - j];

		}
	}
	//save in global memory
	vect_out[ij] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function: Bitwise Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_bitwise_kernel_shfl_xor_SM_Sbox(unsigned long long int *vect, unsigned long long int *vect_out, int sizefor, int sizefor1)
{
	//declaration for shared memory 
	extern __shared__ unsigned long long int tmpsdata1[];

	unsigned int tid = threadIdx.x;
	unsigned int ij = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long long int value = vect[ij];

	value ^= (value & 12297829382473034410) >> 1;
	value ^= (value & 14757395258967641292) >> 2;
	value ^= (value & 17361641481138401520) >> 4;
	value ^= (value & 18374966859414961920) >> 8;
	value ^= (value & 18446462603027742720) >> 16;
	value ^= (value & 18446744069414584320) >> 32;

	int f1, r = 1;

	for (int j = 1; j<sizefor; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//printf("j %d, r %d, f1 %d, val: %d, __shfl_:%d\n", j, r, f1, value,__shfl_down(value, j));
		value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor1; j *= 2)
	{
		tmpsdata1[tid] = value;
		__syncthreads();

		if ((tid&j) == j)
		{
			value = value^tmpsdata1[tid - j];

		}
	}
	//save in global memory
	vect_out[ij] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function: Bitwise Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_bitwise_kernel_shfl_xor_SM_MP(unsigned long long int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory 
	extern __shared__ unsigned long long int tmpsdata1[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (i - (i / fsize)*fsize) * 1024 + (i / fsize); //laneId%n

	// Seed starting value as inverse lane ID 
	unsigned long long int value = VectorValue[ji];
	int f1, r = 1;

	for (int j = 1; j<fsize1; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//printf("j %d, r %d, f1 %d, val: %d, __shfl_:%d\n", j, r, f1, value,__shfl_down(value, j));
		value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<fsize; j *= 2)
	{
		tmpsdata1[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata1[tid - j];
		}
		__syncthreads();
	}
	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Power integer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void powInt(int *Vec, int exp)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	int base = Vec[i];

	__syncthreads();
	int result = 1;

	while (exp)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		base *= base;
	}

	Vec[i] = result;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Algebraic degree
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AD(int *Vec)
{
	int ones = 0;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int value = Vec[i];

	ones = __popc(i)*value;

	Vec[i] = ones;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Bitwise Algebraic degree
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bitwise_AD(unsigned long long int *NumIntVec, int *Vec_max_values, int NumOfBits)
{
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	unsigned int ii = i*NumOfBits;

	unsigned int ones = 0, max = 0;
	int c;
	bool bit;

	unsigned long long int k = 0;
	unsigned long long int number = NumIntVec[i];

	for (c = NumOfBits - 1; c >= 0; c--)
	{
		k = number >> c;

		if (k & 1)
		{
			bit = 1;
			ii++;
		}
		else
		{
			bit = 0;
			ii++;
		}

		ones = __popc(ii - 1)*bit;

		if (max<ones)
			max = ones;
	}
	//save in global memory
	Vec_max_values[i] = max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//*************************************************************************************************************
//Function for S-box
//*************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Algebraic degree Sbox
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AD_Sbox(int *Vec)
{
	int ones = 0;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;

	unsigned int value = Vec[i];

	ones = __popc(tid)*value;

	Vec[i] = ones;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Bitwise Algebraic degree Sbox
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bitwise_AD_Sbox(unsigned long long int *NumIntVec, int *Vec_max_values, int NumOfBits)
{
	int ones = 0, max=0;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	unsigned int ii = tid*NumOfBits;

	unsigned long long int number = NumIntVec[i], k=0;

	int c;
	bool bit;

	for (c = NumOfBits - 1; c >= 0; c--)
	{
		k = number >> c;

		if (k & 1)
		{
			bit = 1;
		}
		else
		{
			bit = 0;
		}

		ones = __popc(ii)*bit;

		ii++;

		if (max<ones)
			max = ones;
	}

	Vec_max_values[i] = max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Component Function of S-box (CF) - first function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnAll_kernel(int *Sbox_in, int *CF_out, int n)
{
	//unsigned int ones=0;
	//ones = __popc (Vect[i]); //Count the number of bits that are set to 1 in a 32 bit integer.
	//Vect[i]=ones;

	int logI, ones, element = 0;

	unsigned int blok = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*n + threadIdx.x;

	int value = Sbox_in[tid];
	logI = value&blok;
	ones = __popc(logI);

	element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//*************************************************************************************************************
//Function forS-box
//*************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Component Function - Polarity True Table of S-box (CF - PTT) - first function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnAll_kernel_PTT(int *Sbox_in, int *CF_out, int n)
{
	//unsigned int ones=0;
	//ones = __popc (Vect[i]); //Count the number of bits that are set to 1 in a 32 bit integer.
	//Vect[i]=ones;

	int logI, ones, element = 0;

	unsigned int blok = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*n + threadIdx.x;

	int value = Sbox_in[tid];
	logI = value&blok;
	ones = __popc(logI);

	//element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)
	//element = 1 - (2 * element);
	element = 1 - (2 * (ones&(1))); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//*************************************************************************************************************
//Function forS-box
//*************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Component Function of S-box (CF) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnVec_kernel(int *Sbox_in, int *CF_out, int row)
{
	//@@ Local variable	
	int logI, ones, element = 0;

	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int value = Sbox_in[i]; //copy data in local variable
	//logI=Vect[tid]&blok;
	logI = value&row;
	//printf("%d ", logI );
	ones = __popc(logI);

	element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//*************************************************************************************************************
//Function forS-box
//*************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Component Function - Polarity True Table of S-box (CF-PTT) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnVec_kernel_PTT(int *Sbox_in, int *CF_out, int row)
{
	//@@ Local variable	
	int logI, ones, element = 0;

	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int value = Sbox_in[i]; //copy data in local variable
	//logI=Vect[tid]&blok;
	logI = value&row;
	//printf("%d ", logI );
	ones = __popc(logI);

	//element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)
	//element = 1 - (2 * element);
	element = 1 - (2 * (ones&(1))); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//*************************************************************************************************************
//Function forS-box
//*************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Difference Distribution Table (DDT) - first function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DDTFnAll_kernel(int *Sbox_in, int *DDT_out, int n)
{
	extern __shared__ int tmpSbox[];

	unsigned int x2, dy;
	unsigned int x1 = threadIdx.x;// & 0x1f;
	unsigned int dx = blockIdx.x;
	//unsigned int i = blockIdx.x*BLOCK_SIZE+threadIdx.x;

	tmpSbox[x1] = Sbox_in[x1];
	__syncthreads();
	x2 = x1 ^ dx;
	// dy = (sbox[x1] ^ sbox[x2])+ blockIdx.x*size;
	dy = (tmpSbox[x1] ^ tmpSbox[x2]) + blockIdx.x*n;
	//atomicInc(&ddt[dy], ddt);
	atomicAdd(&DDT_out[dy], 1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Difference Distribution Table (DDT) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//__global__ void ddtVec_kernel(int *sbox, int *diff_table, int dx)
__global__ void DDTFnVec_kernel(int *Sbox_in, int *DDT_out, int row)
{
	unsigned int x1 = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	int x2, dy;// , value;

	x2 = x1^row; //row - dx

	dy = Sbox_in[x1] ^ Sbox_in[x2];
	atomicAdd(&DDT_out[dy], 1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////