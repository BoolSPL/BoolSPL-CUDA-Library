# BoolSPL: A library with parallel algorithms for Boolean functions and S-boxes for GPU

BoolSPL is open source project for study S-boxes 

#### Current release: v0.2

### What is BoolSPL?
BoolSPL (Boolean S-box parallel library for GPU) provides, reusable software components for every layer of the CUDA programming model [5]. BoolSPLG is a library consisting procedures for analysis and compute cryptographic properties of Boolean and Vector Boolean function (S-box). Our procedures have function for auto grid conﬁguration. Most of the functions are designed to compute the data in registers because they oﬀer the highest bandwidth. 

### Overview of BoolSPLG Basic Procedures 
 The proposed library implement algorithm as composition of basic function into one parameterized kernel, without care about details of implementation. The building function can be classiﬁed into computation (Butterﬂy (FWT, IFWT, FMT, bitwise FMT, min-max), DDT, AlgebraicDegree, ComponentFunction, PowerInt), reordering operations (Copy, MemoryPatern) and support operation reduction (min, max).

Figure 1 presents a scheme with the classiﬁcation of the functions used to build procedures for computing the cryptographic properties of Boolean and Vector Boolean function. The solid line indicates a dependency while the dashed line represents an optional component.

![header image](https://github.com/BoolSPL/BoolSPL-CUDA-Library/blob/master/DiagramBoolSPL.jpg)

##### Figure 1. Classiﬁcation and module dependencies of the building blocks involved in the library

### Functionalities

#### Our library contains the following butterﬂy algorithms: 
- Binary Fast Walsh Transforms (FWT);
- Binary Inverse Fast Walsh Transforms (IFWT);
- Binary Fast Mobius Transforms (FMT);
- Bitwise binary Fast Mobius Transform (bitwise FMT);
- Butterfly Min-Max. 
#### Additional algorithms and function for computing:
- Differential Distribution Tables (DDT);
- Linear Approximation Tables (LAT); 
- Algebraic Normal Forms (ANF);
- Component Function; 
- Auxiliary function reduction for maintaining necessary operations ﬁgure 1.

### Evaluation (cryptographic properties): 
#### Boolean function
- Walsh spectra Wf(f);
- Linearity Lin(f);
- Autocorrelation Spectrum rf(f);
- Autocorrelation AC(f);
- Algebraic Normal Form ANF(f);
- Algebraic Degree Deg(f);


#### Vector Boolean function(s) (S-boxes)
- Linear Approximation Tables LAT(S);
- Linearity Lin(S);
- Autocorrelation spectrum ACT(S);
- Autocorrelation AC(S);
- Algebraic Normal Form ANF(S); 
- Algebraic Degree Deg(S);
- Difference Distribution Table DDT(S);
- Differential uniformity δ;
- Component function Sb. 

### Setup BoolSPL library?

BoolSPL is implemented as a C++ header library. There is no need to “build” BoolSPL separately. To use BoolSPL primitives in your code, simply:

1. Download and unzip the latest BoolSPL distribution from the Downloads section and extract the contents of the zip file to a directory. You need to install (copy) only “BoolSPL” directory from the main BoolSPL-vx.x directory. We suggest installing BoolSPL to the CUDA include directory, which is usually:

   - /usr/local/cuda/include/ on a Linux and Mac OSX;
   - C:\CUDA\include\ on a Windows system.

Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\;

If you are unable to install BoolSPL to the CUDA include directory, then you can place BoolSPL somewhere in your home directory, for example: /home/nathan/libraries/.

2. #include the "umbrella" header file in your CUDA C++ sources;

3. Compile your program with NVIDIA's nvcc CUDA compiler, specifying a -I include-path flag to reference the location of the BoolSPL header library. 
BoolSPL is implemented as a C++ header library. There is no need to “build” BoolSPL separately. To use BoolSPL primitives in your code, simply:

### Examples

BoolSPL distribution directory contain “examples” directory with examples (Boolean and S-box) programs. For the examples to work there is need to include (add) the additional header files from the directory “help and additional header files”. This additional header files contain CPU Boolean and S-box function used for comparison and checking the obtained results from GPU functions.

#### For more read: About BoolSPL - "docs" directory.

#### BoolSPL v0.2 new features (additional functions and procedures), read file "change.log".

### BSbox-tools: console application program

BSbox-tools is developed console application program for representation, defining and computing the most important cryptographic properties of Boolean and Vector Boolean functions (S-boxes). With other words BSbox-tools represents a console interface program on the BoolSPL library. The version BSbox-tools_v0.2 is based on BoolSPL_v0.2 library version. Availaible on: <a href="http://www.moi.math.bas.bg/moiuser/~data/Results/Crypto/BSbox-tools.html">Link</a>

### Reference and Publications related to the BoolSPL library

[1] D. Bikov and I. Bouyukliev, BoolSPLG: A library with parallel algorithms for Boolean functions and S-boxes for GPU, preprint.

[2] D. Bikov, I. Bouyukliev, Parallel Fast Walsh Transform Algorithm and its implementation with CUDA on GPUs. Cybernetics and Information Technologies. Cybernetics and Information Technologies 18, 21–43 (2018). http://www.cit.iit.bas.bg/CIT_2018/v-18-5s/04_paper.pdf

[3] D. Bikov and I. Bouyukliev, Parallel Fast Mobius (Reed-Muller) Transform and its Implementation with CUDA on GPUs, Proceedings of PASCO 2017, Kaiserslautern, Germany, Germany — July 23 - 24, 2017, ISBN: 978-1-4503-5288-8 (improvement presented in this publication are implemented in v0.2 BoolSPL library) https://dlp5.acm.org/citation.cfm?id=3115941

[4] D. Bikov and I. Bouyukliev, BoolSPLG: A library with parallel algorithms for Boolean functions and S-boxes for GPU, Poster session, PUMPS+AI 2018, Barcelona, Spain.

[5] CUDA homepage, Availaible on: https://developer.nvidia.com/cuda-zone

### Additional - Reference and Publications related to the BoolSPL library

[1] I. Bouyukliev, D, Bikov, Applications of the binary representation of integers in algorithms for boolean functions, Proceedings of the Forty Fourth Spring Conference of the Union of Bulgarian Mathematicians SOK “Kamchia”, (2015), pp.161-166, ISSN: 1313-3330 https://core.ac.uk/download/pdf/149219587.pdf

[2] D. Bikov, I. Bouyukliev, Walsh Transform Algorithm and its Parallel Implementation with CUDA on GPUs, Proceedings of 25 YEARS FACULTY OF MATHEMATICS AND INFORMATICS, Veliko Tarnovo, Bulgaria, (2015), pp. 29-34, ISBN: 978-619-00-0419-6

[3] D. Bikov, I. Bouyukliev, A. Stojanova, Beneﬁt of Using Shared Memory in Implementation of Parallel FWT Algorithm with CUDA C on GPUs, Proceedings of 7th International Conference Information Technologies and Education Development, Zrenjanin, Serbia, (2016) pp.250-256, ISBN 978-86-7672-285-3

[4] I. Bouyukliev, D. Bikov, S. Bouyuklieva, S-Boxes from Binary Quasi-Cyclic Codes, Electronic Notes in Discrete Mathematics Volume 57, (2017), pp. 67–72 https://www.sciencedirect.com/science/article/abs/pii/S1571065317300124

[5] D. Bikov, I. Bouyukliev and S. Bouyuklieva, 2019. Bijective S-boxes of different sizes obtained from quasi-cyclic codes. Journal of Algebra Combinatorics Discrete Structures and Applications, 6(3), pp.123-134. http://jm.jacodesmath.com/index.php/jacodesmath/article/view/212

#### If you have any questions or comments, please do not hesitate to email at dusan.bikov@ugd.edu.mk or iliyab@math.bas.bg
