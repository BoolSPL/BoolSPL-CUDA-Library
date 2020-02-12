//Help Heder file "func_Boolean_CPU.h" - CPU compute Boolean functions properties
//System includes
#include <stdio.h>
#include <iostream>
#include <algorithm>

using namespace std;

///////////////////////////////////////////////////////////////////////////
//declaration of function
///////////////////////////////////////////////////////////////////////////
int reduceCPU_max(int *vals, int nvals);

void FastWalshTrans(int n, int *BoolSbox, int *walshvec);
void FastWalshTransInv(int n, int *walshvec);
void FastMobiushTrans(int size, int *TT, int *ANF);

int ipow(int base, int exp);
void PTT_fun_pow2(int size, int *vec);

void Fill_dp_vector(int n, int *Vect);
void check_rez(int *Vec1, int *Vec2, int size);
void Print_Result(int n, int *Result, int size);

int FindMaxDeg(int size, int *ANF_CPU);
///////////////////////////////////////////////////////////////////////////

//====== CPU function find max absolute value in array ========================================================
int reduceCPU_max(int *vals, int nvals)
{
	//int cmax = vals[0];

	int cmax = vals[0];

	for (int i = 1; i<nvals; i++)
	{
		cmax = max(abs(vals[i]), cmax);
	}
	return cmax;
}
//=============================================================================================================

//====== CPU function find max absolute value in array AC(f) ==================================================
int reduceCPU_AC(int nvals, int *AC)
{
	AC[0] = 0;
	int cmax = AC[1];

	for (int i = 2; i<nvals; i++)
	{
		cmax = max(abs(AC[i]), cmax);
	}
	return cmax;
}
//=============================================================================================================

//============ CPU Fast Walsh Transform function ==============================================================
void FastWalshTrans(int n, int *BoolSbox, int *walshvec)
{
	int temp = 0, j = 1;
	for (int i = 0; i<n; i++)
	{
		walshvec[i] = 0;
		walshvec[i] = BoolSbox[i];
	}

	while (j<n)
	{
		for (int i = 0; i<n; i++){
			if ((i&j) == 0)
			{
				temp = walshvec[i];

				walshvec[i] = (walshvec[i] + walshvec[i + j]);
				walshvec[i + j] = (-walshvec[i + j] + temp);
			}
		}
		j = j * 2;
	}
}
//=============================================================================================================

//============ CPU Invers Fast Walsh Transform function =======================================================
void FastWalshTransInv(int n, int *walshvec)
{
	int temp = 0, j = 1;
	//	for(int i=0; i<n; i++)
	//{
	//	walshvec[i]=0;
	//	walshvec[i]=BoolSbox[i];
	//}

	while (j<n)
	{
		for (int i = 0; i<n; i++)
		{
			if ((i&j) == 0)
			{
				temp = walshvec[i];

				walshvec[i] = (walshvec[i] + walshvec[i + j]) / 2;
				walshvec[i + j] = (-walshvec[i + j] + temp) / 2;
			}
		}
		j = j * 2;
	}
}
//=============================================================================================================

//============= CPU Fast Mobius Transform function ============================================================
void FastMobiushTrans(int size, int *TT, int *ANF)
{
	int j = 1;
	for (int i = 0; i<size; i++)
	{
		ANF[i] = 0;
		ANF[i] = TT[i];
	}

	while (j<size)
	{
		for (int i = 0; i<size; i++)
		{
			if ((i&j) == j)
			{
				ANF[i] = ANF[i] ^ ANF[i - j];
				//SALg[jj][i]=SALg[jj][i]^SALg[jj][i-j];
			}
		}
		j = j * 2;
	}
}
//=============================================================================================================

//============== CPU Power function ===========================================================================
int ipow(int base, int exp)
{
	int result = 1;
	while (exp)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		base *= base;
	}

	return result;
}
//=============================================================================================================

//============== CPU Power 2 array function ===================================================================
void PTT_fun_pow2(int size, int *vec)
{
	for (int j = 0; j<size; j++)
	{
		vec[j] = ipow(vec[j], 2);

	}
}
//=============================================================================================================

//================ CPU random array fill function =============================================================
void Fill_dp_vector(int n, int *Vect, int *VectP)
{
	for (int j = 0; j<n; j++)
	{
		Vect[j] = rand() % 2;

		if (Vect[j] == 0)
			VectP[j] = 1;
		else
			VectP[j] = -1;
		//Vect[j]=1;
	}
}
//=============================================================================================================

//============== CPU function for check two input arrays and print result =====================================
void check_rez(int *Vec1, int *Vec2, int size)
{
	bool check = true;
	for (int i = 0; i<size; i++)
	{
		if (Vec1[i] != Vec2[i])
		{
			check = false;
			break;
		}
	}

	if (check)
		cout << "\nCheck: True\n";

	else
		cout << "\nCheck: False\n";
}
//=============================================================================================================

//============== CPU function for check two input arrays and return result ====================================
int check_rez_return(int *Vec1, int *Vec2, int size)
{
	bool check = true;
	for (int i = 0; i<size; i++)
	{
		if (Vec1[i] != Vec2[i])
		{
			check = false;
			break;
		}
	}

	if (check)
		return 1;

	else
		return 0;
}
//=============================================================================================================

//=================== CPU function print array ================================================================
void Print_Result(int n, int *Result, int size)
{
	for (int j = 0; j<size; j++)
	{
		printf("%d, ", Result[j]);
	}
	printf("\n\n");
}
//=============================================================================================================

//=================== CPU function find max algebraic degree ==================================================
int FindMaxDeg(int size, int *ANF_CPU)
{
	unsigned int ones = 0, max = 0;// , min = 100;
	for (int i = 0; i<size; i++)
	{
		ones = _mm_popcnt_u32(i)*ANF_CPU[i];
		if (max<ones)
			max = ones;
		//if ((min>ones)&(ones != 0))
		//	min = ones;

		//cout << ones << " ";
	}
	return max;
}
//=============================================================================================================

//================ Bitwise Fast Mobiush Transform ====================
void FastMobiushTransBitwise(int size, unsigned long long int *TT)
{
	int j = 1;

	while (j<size)
	{
		for (int i = 0; i<size; i++)
		{
			if ((i&j) == j)
			{
				TT[i] = TT[i] ^ TT[i - j];
				//SALg[jj][i]=SALg[jj][i]^SALg[jj][i-j];
			}
		}
		j = j * 2;
	}
}

//========== start CPU bitwise function ================================
void CPU_FWT_bitwise(unsigned long long int *NumIntVec, unsigned long long int *NumIntVecANF, int NumOfBits, int NumInt)
{

	for (int j = 0; j<NumInt; j++)
	{
		unsigned long long  int brez = NumIntVec[j];// , counter1 = 0;
		//unsigned long long int  bshift, b;

		brez ^= (brez & 12297829382473034410) >> 1;
		brez ^= (brez & 14757395258967641292) >> 2;
		brez ^= (brez & 17361641481138401520) >> 4;
		brez ^= (brez & 18374966859414961920) >> 8;
		brez ^= (brez & 18446462603027742720) >> 16;
		brez ^= (brez & 18446744069414584320) >> 32;

		NumIntVecANF[j] = brez;

		//cout << "\nCPU rez:" << brez << "\n";
	}

	if (NumInt>1)
		FastMobiushTransBitwise(NumInt, NumIntVecANF);

	//cout << "\nPrint Integer/s ANF (CPU bitwise):";
	//for(int i=0; i<NumInt; i++)
	//cout <<NumIntVecANF[i] << " ";
	//cout << "\n\n";
}

////============= CPU function for set TT in 64 bit int variables ==========================
//void BinVecToDec(int size, int *Bin_Vec, unsigned long long int *NumIntVec, int NumInt)
//{
//	for (int i = 0; i<NumInt; i++)
//	{
//		unsigned long long int decimal = 0, sum = 0, bin, counterBin = 0;
//		int set = i*size;
//		//cout << "Set:" << set ;
//		for (int j = ((size - 1) + set); j >= (0 + set); j--)
//		{
//			bin = Bin_Vec[j];
//			decimal = bin << counterBin;
//			counterBin++;
//			sum = sum + decimal;
//		}
//		NumIntVec[i] = sum;
//		//cout << "Number:"<< sum << "\n";
//	}
//}
////=================================================================================

////============= CPU function for set ANF from NumIntVector ==========================
//void DecVecToBin(int NumOfBits, int *Bin_Vec, unsigned long long int *NumIntVec, int NumInt)
//{
//	unsigned long long int number = 0;
//	int c, k, ii = 0;
//
//	for (int i = 0; i<NumInt; i++)
//	{
//		number = NumIntVec[i];
//
//		for (c = NumOfBits - 1; c >= 0; c--)
//		{
//			k = number >> c;
//
//			if (k & 1)
//			{
//				Bin_Vec[ii] = 1;
//				//cout << Bin_Vec[ii] << " ";
//				ii++;
//			}
//			else
//			{
//				Bin_Vec[ii] = 0;
//				//cout << Bin_Vec[ii] << " ";
//				ii++;
//			}
//		}
//		//cout << "Number:"<< sum << "\n";
//	}
//}
////=================================================================================

//============= CPU function for set ANF from NumIntVector ==========================
int DecVecToBin_maxDeg(int NumOfBits, unsigned long long int *NumIntVec, int NumInt)
{
	unsigned int ones = 0, max = 0;// , min = 100;

	unsigned long long int number = 0, k = 0;
	int c, ii = 0;

	bool bit;

	for (int i = 0; i<NumInt; i++)
	{
		number = NumIntVec[i];
		max = 0, k = 0;

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

			ones = _mm_popcnt_u32(ii - 1)*bit;

			if (max<ones)
				max = ones;
		}
		//cout << max << " ";
	}
	return max;
}
//=================================================================================