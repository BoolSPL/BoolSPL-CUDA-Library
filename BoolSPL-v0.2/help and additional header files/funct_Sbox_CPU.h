//Help Heder file "funct_Sbox_CPU.h" - CPU computing S-box functions properties
// System includes
#include <stdio.h>
#include <iostream>
#include <algorithm>

//declare global vector
int *PTT, *TT, *WHT, *t, *ANF; // SboxDec, *;

using namespace std;

//========= function Header Compute Properties ===================
void HeaderCompProperties(int size, int *SboxDec, int bin, int **STT)
{
	//================================================================
	int counterSboxVer = 0; //counter for verification of linearity of whole S-box

	int m = bin - 1;

	for (int e = 1; e <= m; e++)
		t[e] = e + 1;

	for (int j = 0; j<size; j++)
	{
		TT[j] = 0;
	}

	int i = 1;

	while (i != m + 1)
	{
		for (int j = 0; j<size; j++)
		{

			TT[j] = TT[j] ^ STT[i][j];

			if (TT[j] == 1)
				PTT[j] = -1;
			else
				PTT[j] = 1;
		}
		t[0] = 1;
		t[i - 1] = t[i];
		t[i] = i + 1;
		i = t[0];

		//Function: Fast Walsh Transformation function CPU
		FastWalshTrans(size, PTT, WHT);	//Find Walsh spectra on one row
		//Lin_cpu = reduceCPU_PTT(size, PTT);		//Find Linearity on one row //max value from Walsh spectra
		Lin_return = reduceCPU_max(WHT, size);

		if (Lin_cpu < Lin_return)
			Lin_cpu = Lin_return;
		
		//Function: Fast Mobiush Transformation function CPU
		FastMobiushTrans(size, TT, ANF);
		AD_return = AlgDegMax(ANF, size);

		if (ADmax_cpu < AD_return)
			ADmax_cpu = AD_return;

		//Function: Autocorelation Transformation function CPU
		PTT_fun_pow2(size, WHT);
		FastWalshTransInv(size, WHT);
		AC_return = reduceCPU_AC(size, WHT);
		//AC_all[counterSboxVer] = ACn_cpu;
		if (ACn_cpu < AC_return)
			ACn_cpu = AC_return;

		//***** counter *****
		counterSboxVer++;

		//*********** function for DDT *********
		DDT_vector(SboxDec, counterSboxVer);
		//**************************************
	}

	//Lin_cpu = reduceCPU_PTT_All(size - 1, PTT_ALL_LIN); //Find Lin of S-box
	//Lin_cpu = reduceCPU_max(PTT_ALL_LIN, size - 1);
	nl_cpu = sizeSbox / 2 - (Lin_cpu / 2);		//Compute Nonlinearity

	cout << "\nLinearity(Lin):" << Lin_cpu << "\n";
	cout << "Nonlinearity(nl):" << nl_cpu << "\n";
	cout << "Differential uniformity(Diff):" << Diff_cpu << "\n";
	cout << "Autocorrelation(AC): " << ACn_cpu << "\n";
	cout << "Alg. Deagree (max):" << ADmax_cpu << "\n";
	//================================================================
}

///////////////////////////////////////////////////////////////////////////
//declaration of function
///////////////////////////////////////////////////////////////////////////

//============ main function for compute Sbox properties ===========
void MainSboxProperties(int **STT, int *SboxDec)
{
	//Allocate memory blocks
	PTT = (int *)malloc(sizeof(int)* sizeSbox);
	TT = (int *)malloc(sizeof(int)* sizeSbox);
	t = (int *)malloc(sizeof(int)* binary);

	WHT = (int *)malloc(sizeof(int)* sizeSbox);
	ANF = (int *)malloc(sizeof(int)* sizeSbox);

	Lin_cpu = 0, nl_cpu = 0, Diff_cpu = 0, ACn_cpu = 0, ADmax_cpu = 0;

	HeaderCompProperties(sizeSbox, SboxDec, binary, STT);

	//============== free memory ==========================
	free(PTT);
	free(TT);
	free(t);

	free(ANF);
	free(WHT);
}
//==================================================================