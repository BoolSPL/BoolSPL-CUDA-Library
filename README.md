# BoolSPL: A library with parallel algorithms for Boolean functions and S-boxes for GPU

BoolSPL is open source project for study S-boxes 

### What is BoolSPL?
BoolSPL (Boolean S-box parallel library for GPU) provides, reusable software components for every layer of the CUDA programming model [5]. BoolSPLG is a library consisting procedures for analysis and compute cryptographic properties of Boolean and Vector Boolean function (S-box). Our procedures have function for auto grid conﬁguration. Most of the functions are designed to compute the data in registers because they oﬀer the highest bandwidth. 

### Overview of BoolSPLG Basic Procedures 
 The proposed library implement algorithm as composition of basic function into one parameterized kernel, without care about details of implementation. The building function can be classiﬁed into computation (Butterﬂy (FWT, IFWT, FMT, bitwise FMT, min-max), DDT, AlgebraicDegree, ComponentFunction, PowerInt), reordering operations (Copy, MemoryPatern) and support operation reduction (min, max).

Figure 1 presents a scheme with the classiﬁcation of the functions used to build procedures for computing the cryptographic properties of Boolean and Vector Boolean function. The solid line indicates a dependency while the dashed line represents an optional component.

![header image](https://github.com/BoolSPL/BoolSPL-CUDA-Library/blob/master/DiagramBoolSPL.jpg)

#### Figure 1. Classiﬁcation and module dependencies of the building blocks involved in the library
