//Help Heder file "helpSboxfunct.h" - CPU computing S-box functions properties and other functions
//input header files
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <nmmintrin.h>

using namespace std;

//============== Function set Binary ==============================
void SetBinary()
{
	int binar = 1;
	while (binar<sizeSbox)
	{
		binar = binar * 2;
		binary++;
	}
	binary++;
}
//=================================================================

//======Function set parametar Size of S-box and Binary============
void SetParSbox(string filename)
{
	vector <string> words; // Vector to hold our words we read in.
	string str; // Temp string to
	ifstream fin(filename); // Open it up!
	if (fin.is_open())
	{
		//cout << " " << fin.getline(str1, 10,'_') << "\n";
		//file opened successfully so we are here
		cout << "File Contain 'S-box' is Opened successfully!.\n";
		while (fin >> str) // Will read up to eof() and stop at every
		{                  // whitespace it hits. (like spaces!)
			words.push_back(str);
			//InvPerm1[counterPerFrile]=atoi(str.c_str());
		}
		fin.close(); // Close that file!

		sizeSbox = words.size();
	}
	else //file could not be opened
	{
		cout << "File '" << filename << "' could not be opened." << endl;
		CheckFile = false;
		return;
	}
	//set number of binary element
	SetBinary();
}
//=================================================================

//================= Read S-box element from file ==================
void readFromFileMatPerm(string filename, int *Element)
{
	int counterPerFrile = 0;
	vector <string> words; // Vector to hold our words we read in.
	string str; // Temp string to
	ifstream fin(filename); // Open it up!
	if (fin.is_open())
	{
		//cout << " " << fin.getline(str1, 10,'_') << "\n";
		//file opened successfully so we are here
		cout << "\nFile '" << filename << "' Opened successfully!.\n";
		while (fin >> str) // Will read up to eof() and stop at every
		{                  // whitespace it hits. (like spaces!)
			words.push_back(str);
			Element[counterPerFrile] = atoi(str.c_str());

			counterPerFrile++;
		}
		fin.close(); // Close that file!
		//int chNumMatrx = counterPerFrile;
		cout << "Number of element into file " << filename << ": " << counterPerFrile;
	}
	else //file could not be opened
	{
		cout << "File '" << filename << "' could not be opened." << endl;
		CheckFile = false;
	}
}
//=================================================================

//============ function for binary convert ========================
void binary1(int number, int *binary_num) {
	int w = number, c, k, i = 0;
	for (c = binary - 1; c >= 0; c--)
	{
		k = w >> c;

		if (k & 1)
		{
			binary_num[i] = 1;
			i++;
		}
		else
		{
			binary_num[i] = 0;
			i++;
		}
	}
}
//=================================================================

//============ Set STT file for Sbox ==============================
void SetSTT(int *SboxElemet, int **STT, int *binary_num)
{
	int elementS = 0;
	for (int j = 0; j<sizeSbox; j++)
	{
		elementS = SboxElemet[j];
		binary1(elementS, binary_num);

		for (int i = 0; i<binary; i++)
		{
			STT[i][j] = binary_num[i];
		}
	}
}
//=================================================================

//====== CPU computing DDT(S) function ============================
void DDT_vector(int *sbox, int dx)
{
	int* diff_table = new int[sizeSbox]();

	int x1, x2, dy;
	for (x1 = 0; x1 < sizeSbox; ++x1) {
		//  for (dx = 0; dx < sbox_size; ++dx) {
		x2 = x1 ^ dx;
		dy = sbox[x1] ^ sbox[x2];
		++diff_table[dy];
		// }
		if (diff_table[dy]>Diff_cpu)
			Diff_cpu = diff_table[dy];
	}
	//  for (int i = 0; i < sizeSbox; ++i) {
	//        std::cout << std::setw(4) << diff_table[i];
	//    }
	delete[] diff_table;
}
//=================================================================

//====== CPU computing max deg(S) function ========================
int AlgDegMax(int *ANF_CPU, int size)
{
	unsigned int ones = 0, max = 0;
	for (int i = 0; i<size; i++)
	{
		ones = _mm_popcnt_u32(i)*ANF_CPU[i];
		if (max<ones)
			max = ones;
		//if((min>ones)&(ones!=0))
		//	min=ones;

		//	cout <<ones<< " ";
	}
	//	cout <<"Alg. Deagree (max):" << max <<" Alg. Deagree (min):" << min <<"\n";	
	return max;
}
//=================================================================

//====== CPU computing min deg(S) function ========================
int AlgDegMin(int *ANF_CPU, int size)
{
	unsigned int ones = 0, min = 100;
	for (int i = 0; i<size; i++)
	{
		ones = _mm_popcnt_u32(i)*ANF_CPU[i];
		//if(max<ones)
		//	max=ones;
		if ((min>ones)&(ones != 0))
			min = ones;

		//	cout <<ones<< " ";
	}
	//cout << min << " ";
	return min;
	//	cout <<"Alg. Deagree (max):" << max <<" Alg. Deagree (min):" << min <<"\n";	
}
//=================================================================

//===== CPU computing component function (CF) of S-box function ===
//===== all CF are save in one array CPU_STT ======================
//void GenTTComponentFunc(int j, int *SboxElemet, int *CPU_STT)
//{
//	unsigned int ones = 0, logI, element;
//
//	for (int i = 0; i<sizeSbox; i++)
//	{
//		logI = SboxElemet[i] & j;
//		// ones = _mm_popcnt_u32(SboxElemet[i]);
//		ones = _mm_popcnt_u32(logI);
//		//cout << ones << " ";
//		// cout << logI << " ";
//		if (ones % 2 == 0)
//			element = 0;
//		else
//			element = 1;
//
//		//cout << element << ", ";
//
//		CPU_STT[j*sizeSbox + i] = element;
//	}
//	//cout << "\n";
//}
//=================================================================

//===== CPU computing component function (CF) of S-box function ===
//===== One CF is save in array CPU_STT ===========================
//void GenTTComponentFuncVec(int j, int *SboxElemet, int *CPU_STT)
//{
//	unsigned int ones = 0, logI, element;
//
//	for (int i = 0; i<sizeSbox; i++)
//	{
//		logI = SboxElemet[i] & j;
//		// ones = _mm_popcnt_u32(SboxElemet[i]);
//		ones = _mm_popcnt_u32(logI);
//		//cout << ones << " ";
//		// cout << logI << " ";
//		if (ones % 2 == 0)
//			element = 0;
//		else
//			element = 1;
//
//		//		cout << element << ", ";
//
//		CPU_STT[i] = element;
//	}
//	//	cout <<"\n";
//}
//=================================================================