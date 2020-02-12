//BoolSPLG Boolean Algorithms
// System includes
#include <stdio.h>
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Max Butterfly function
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Butterfly_max_kernel(int sizeSbox, int *device_data)
{
	CheckSize(sizeSbox);

	int max;

	//@ Set grid
	setgrid(sizeSbox);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_data, device_data, sizethread);
	if (sizeSbox>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_data, sizeblok, sizeblok1);

	cudaMemcpy(&max, &device_data[0], sizeof(int), cudaMemcpyDeviceToHost);

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bool Fast walsh transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		int max;

		//call Reduction Max
		max = runReductionMax(size, device_Vect_rez);
		return max;
	}
	else
	{
		return 0;
	}
	//////////////////////////////////
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bool Fast walsh transform use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		int max = 0;
		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

		cudaMemcpy(&max, &device_Vect_rez[0], sizeof(int), cudaMemcpyDeviceToHost);


		return max;
	}
	else
	{
		return 0;
	}
	//////////////////////////////////
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bool Fast Mobius transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MobiusTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);
	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bool Bitwise Fast Mobius transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BitwiseMobiusTranBoolGPU(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSizeBoolBitwise(size);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = size / NumOfBits;

	//@ Set grid Bitwise
	setgridBitwise(NumInt);

	///////////////////////////////////////////////////
	fmt_bitwise_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect, device_Vect_rez, sizefor, sizefor1);
	if (NumInt>1024)
		fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect_rez, sizeblok, sizeblok1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bool Algebriac degree
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeBoolGPU(int *device_Vect, int *device_Vect_rez, int size)
{
	CheckSize(size);

	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	kernel_AD << <sizeblok, sizethread >> >(device_Vect_rez);

	int max = 0;

	//max = SetRunMaxReduction(size, device_Vect_rez);
	max = runReductionMax(size, device_Vect_rez);

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bool Algebriac degree use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size)
{
	CheckSize(size);

	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	kernel_AD << <sizeblok, sizethread >> >(device_Vect_rez);

	int max = 0;
	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	cudaMemcpy(&max, &device_Vect_rez[0], sizeof(int), cudaMemcpyDeviceToHost);


	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bitwise Bool Algebriac degree use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int *device_Vec_max_values, int *host_max_values, int size)
{
	//Check the input size of Boolean function
	CheckSizeBoolBitwise(size);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = size / NumOfBits, max=0;

	//@ Set grid Bitwise
	setgridBitwise(NumInt);

	///////////////////////////////////////////////////
	fmt_bitwise_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect, device_Vect_rez, sizefor, sizefor1);
	if (NumInt>1024)
		fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect_rez, sizeblok, sizeblok1);
	///////////////////////////////////////////////////

	kernel_bitwise_AD << < sizeblok, sizethread >> >(device_Vect_rez, device_Vec_max_values, NumOfBits);

	if (NumInt > 256)
	{
	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM <<<sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, device_Vec_max_values, sizethread);
	if (NumInt>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP <<< sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, sizeblok, sizeblok1);

	cudaMemcpy(&max, &device_Vec_max_values[0], sizeof(int), cudaMemcpyDeviceToHost);

	return max;
	}
	else
	{
		cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int)* NumInt, cudaMemcpyDeviceToHost);

		max=reduceCPU_max_libhelp(host_max_values, NumInt);

		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Autocorrelation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	CheckSize(size);

	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	powInt << < sizeblok, sizethread >> >(device_Vect_rez, 2);

	ifmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		ifmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		int max;
		cudaMemset(device_Vect_rez, 0, 1 * sizeof(int));
		//	max = SetRunMaxReduction(size, device_Vect_rez);
		max = runReductionMax(size, device_Vect_rez);
		return max;
	}
	else
	{
		return 0;
	}
	//////////////////////////////
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Autocorrelation use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	CheckSize(size);

	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	powInt << < sizeblok, sizethread >> >(device_Vect_rez, 2);

	ifmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		ifmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		int max;
		cudaMemset(device_Vect_rez, 0, 1 * sizeof(int));

		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

		cudaMemcpy(&max, &device_Vect_rez[0], sizeof(int), cudaMemcpyDeviceToHost);

		return max;
	}
	else
	{
		return 0;
	}
	//////////////////////////////
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Linear Aproximation Table (LAT) - Linearity S-box
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranSboxGPU(int *device_Sbox, int *device_CF, int *device_LAT, int sizeSbox)
{
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute LAT of S-box
		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_LAT, sizethread);

		cudaMemset(device_LAT, 0, sizeSbox*sizeof(int)); //clear first row of LAT !!!
		//@Reduction Max return Lin of the S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_LAT);

		return max;
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int *ALL_LAT = (int *)malloc(sizeof(int)* sizeSbox);
		ALL_LAT[0] = 0;

		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			max = WalshSpecTranBoolGPU(device_CF, device_LAT, sizeSbox, true);
			//max = runReductionMax(sizeSbox, device_LAT);
			ALL_LAT[i] = max;
		}

		//cudaMemset(device_LAT, 0, sizeSbox*sizeof(int)); //clear LAT !!!
		// Copy input vectors from host memory to GPU buffers.
		cudaMemcpy(device_LAT, ALL_LAT, sizeof(int)*sizeSbox, cudaMemcpyHostToDevice);
		//@Reduction Max return Lin of the S-box
		max = runReductionMax(sizeSbox, device_LAT);

		//@Free memory
		free(ALL_LAT);
		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Linear Aproximation Table (LAT) use Max Butterfly - Linearity S-box
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_LAT, int sizeSbox, bool returnMax)
{
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute LAT of S-box
		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_LAT, sizethread);

		if (returnMax)
		{
		int max = 0; //max variable
		cudaMemset(device_LAT, 0, sizeSbox*sizeof(int)); //clear first row of LAT !!!

		//use Max Butterfly return Lin of the S-box
		max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_LAT);

		return max;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (returnMax)
		{
			int max = 0, MaxReturn = 0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				//@Compute Component function GPU
				ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);
				if (returnMax)
				{
					MaxReturn = WalshSpecTranBoolGPU_ButterflyMax(device_CF, device_LAT, sizeSbox, true);
					
					if (MaxReturn>max)
						max = MaxReturn;
				}
			}
			return max;
		}
		else 
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Algebraic Normal Form (ANF) - Algebraic Degree
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int MobiusTranSboxADGPU(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox)
{
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ANF, sizethread);

		kernel_AD_Sbox << <sizeblok, sizethread >> >(device_ANF);

		//@Reduction Max return Lin of the S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_ANF);

		return max;
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int *ALL_ANF = (int *)malloc(sizeof(int)* sizeSbox);
		ALL_ANF[0] = 0;

		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			max = AlgebraicDegreeBoolGPU(device_CF, device_ANF, sizeSbox);
			ALL_ANF[i] = max;
		}

		//cudaMemset(device_LAT, 0, sizeSbox*sizeof(int)); //clear LAT !!!
		// Copy input vectors from host memory to GPU buffers.
		cudaMemcpy(device_ANF, ALL_ANF, sizeof(int)*sizeSbox, cudaMemcpyHostToDevice);
		//@Reduction Max return ANF of the S-box
		max = runReductionMax(sizeSbox, device_ANF);

		//@Free memory
		free(ALL_ANF);
		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Algebraic Normal Form (ANF)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MobiusTranSboxGPU(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox)
{
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ANF, sizethread);
	}
	else
	{
		cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Algebraic Normal Form (ANF) Bitwise
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BitwiseMobiusTranSboxGPU(int *host_Sbox, int *host_Vect_CF, unsigned long long int *host_NumIntVecCF, unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int sizeSbox)
{
	CheckSizeSboxBitwiseMobius(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = 0;

	if (sizeSbox<16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		int sizefor, sizefor1;
		NumInt = (sizeSbox*sizeSbox) / NumOfBits;
		//Compute Component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			// CPU computing component function (CF) of S-box function - all CF are save in one array 
			GenTTComponentFunc(i, host_Sbox, host_Vect_CF, sizeSbox);
		}

		//convert bool into integers
		BinVecToDec(NumOfBits, host_Vect_CF, host_NumIntVecCF, NumInt);

		cudaMemcpy(device_NumIntVecCF, host_NumIntVecCF, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

		//Set Bitwise S-box GRID
		if (sizeSbox < 2048)
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = sizeSbox / 64;
			sizefor1 = 32;
		}
		else
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = 32;
			sizefor1 = NumInt / NumOfBits;
		}

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

	}
	////////////////////////////////////////////////////////
	else
	{
		cout << "\nIs not implemented this funcionality for S-box size n>14. \nThe output data is to big.\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Algebraic Degree use Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//int WalshSpecTranSboxGPU(int *device_Sbox, int *device_CF, int *device_LAT, int sizeSbox)
int AlgebraicDegreeSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox)
{
	CheckSizeSbox(sizeSbox);

	int max = 0, returnMax=0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ANF, sizethread);

		kernel_AD_Sbox << <sizeblok, sizethread >> >(device_ANF);

		//use Max Butterfly return ANF of the S-box
		max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_ANF);

		return max;
	}
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			returnMax = AlgebraicDegreeBoolGPU_ButterflyMax(device_CF, device_ANF, sizeSbox);
			
			if (returnMax>max)
				max = returnMax;
		}
		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bitwise Algebraic Degree use Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//int WalshSpecTranSboxGPU(int *device_Sbox, int *device_CF, int *device_LAT, int sizeSbox)
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMax(int *host_Sbox, int *host_Vect_CF, int *host_max_values, unsigned long long int *host_NumIntVecCF, unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int *device_Vec_max_values, int sizeSbox)
{
	CheckSizeSboxBitwise(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = 0, max = 0, returnMax = 0; //max variable
	
	if (sizeSbox<16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		int sizefor, sizefor1;
		NumInt = (sizeSbox*sizeSbox) / NumOfBits;
		//Compute Component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			// CPU computing component function (CF) of S-box function - all CF are save in one array 
			GenTTComponentFunc(i, host_Sbox, host_Vect_CF, sizeSbox);
		}

		//convert bool into integers
		BinVecToDec(NumOfBits, host_Vect_CF, host_NumIntVecCF, NumInt);

		cudaMemcpy(device_NumIntVecCF, host_NumIntVecCF, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

		//Set Bitwise S-box GRID
		if (sizeSbox < 2048)
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = sizeSbox / 64;
			sizefor1 = 32;
		}
		else
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = 32;
			sizefor1 = NumInt / NumOfBits;
		}

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> >(device_NumIntVecANF, device_Vec_max_values, NumOfBits);

		if (NumInt > 256)
		{
			//@ Set grid
			setgrid(NumInt);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, device_Vec_max_values, sizethread);
			if (NumInt>1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, sizeblok, sizeblok1);

			cudaMemcpy(&max, &device_Vec_max_values[0], sizeof(int), cudaMemcpyDeviceToHost);	

			return max;
		}
		else
		{
		cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int)* NumInt, cudaMemcpyDeviceToHost);

		max = reduceCPU_max_libhelp(host_max_values, NumInt);

			return max;
		}
	}
	////////////////////////////////////////////////////////
	else
	{
		NumInt = sizeSbox / NumOfBits;
		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== One CF is save in array CPU_STT ===========================
			GenTTComponentFuncVec(i, host_Sbox, host_Vect_CF, sizeSbox);

			//convert bool into integers
			BinVecToDec(NumOfBits, host_Vect_CF, host_NumIntVecCF, NumInt);

			cudaMemcpy(device_NumIntVecCF, host_NumIntVecCF, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

			returnMax = BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(device_NumIntVecCF, device_NumIntVecANF, device_Vec_max_values, host_max_values, sizeSbox);
			
			if (returnMax > max)
				max = returnMax;
		}
		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Autocorrelation Transform (ACT) - Autocorrelation (AC)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranSboxGPU(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox)
{
	CheckSizeSbox(sizeSbox);

	int max = 0, returnMax=0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ACT of S-box

		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ACT, sizethread);

		powInt << < sizeblok, sizethread >> >(device_ACT, 2);

		ifmt_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_ACT, device_ACT, sizethread);
			
		cudaMemset(device_ACT, 0, sizeSbox*sizeof(int)); //clear first row of ACT !!!
		//@Reduction Max return ACT of the S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_ACT);

	}
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Return AC(S) of component function of the S-box
			returnMax = AutocorrelationTranBoolGPU(device_CF, device_ACT, sizeSbox, true);

			if (returnMax>max)
				max = returnMax;
		}

	}
	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Autocorrelation Transform (ACT) use Max Butterfly - Autocorrelation (AC)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox, bool returnMax)
{
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ACT of S-box

		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ACT, sizethread);

		powInt << < sizeblok, sizethread >> >(device_ACT, 2);

		ifmt_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_ACT, device_ACT, sizethread);

		if (returnMax)
		{
			int max = 0; //max variable

			cudaMemset(device_ACT, 0, sizeSbox*sizeof(int)); //clear first row of ACT !!!

			//use Max Butterfly return ACT of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_ACT);

			return max;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (returnMax)
		{
			int max = 0, MaxReturn=0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				//@Compute Component function GPU
				ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

				//@Return AC(S) of component function of the S-box
				MaxReturn = AutocorrelationTranBoolGPU_ButterflyMax(device_CF, device_ACT, sizeSbox, true);

				if (MaxReturn>max)
					max = MaxReturn;
			}

			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Difference Distribution Table (DDT) - Differential uniformity
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DDTSboxGPU(int *device_Sbox, int *device_DDT, int sizeSbox)
{
	CheckSizeSbox(sizeSbox);

	int max = 0, maxReturn = 0; //max variable

	if (sizeSbox <= BLOCK_SIZE)
	{
		//set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		cudaMemset(device_DDT, 0, sizeSbox*sizeSbox*sizeof(int));
		DDTFnAll_kernel << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Sbox, device_DDT, sizeSbox);

		//return Max reduction

		cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear first row of DDT !!!
		max = runReductionMax(sizeSbox*sizeSbox, device_DDT);

		return max;
	}
	//////////////////////////////////////////////////////
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear DDT !!!
			DDTFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_DDT, i);
			maxReturn = runReductionMax(sizeSbox, device_DDT);

			if (maxReturn > max)
				max = maxReturn;
		}

		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Difference Distribution Table (DDT) use Max Butterfly - Differential uniformity
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DDTSboxGPU_ButterflyMax(int *device_Sbox, int *device_DDT, int sizeSbox, bool returnMax)
{
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		cudaMemset(device_DDT, 0, sizeSbox*sizeSbox*sizeof(int));
		DDTFnAll_kernel << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Sbox, device_DDT, sizeSbox);

		if (returnMax)
		{
			int max = 0; //max variable
			cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear first row of DDT !!!

			//use Max Butterfly return delta of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_DDT);

			return max;
		}
		else
		{
			return 0;
		}
	}
	//////////////////////////////////////////////////////
	else
	{
		if (returnMax)
		{
			int max = 0, maxReturn=0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear DDT !!!
				DDTFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_DDT, i);
				
				//use Max Butterfly return delta of the S-box component function
				maxReturn = Butterfly_max_kernel(sizeSbox, device_DDT);

				if (maxReturn > max)
					max = maxReturn;
			}

			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////