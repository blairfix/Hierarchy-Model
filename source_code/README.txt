===========================
HIERARCHY MODEL SOURCE CODE 
===========================

Copyright Blair Fix
blairfix@gmail.com


Code for the hierarchy model is written in C++. Core model functions are contained in "/core" directory.
Utility functions (for analysis) are contained in "/utils" director.



REQUIRED LIBRARIES
==================

To compile this code, you will need to install the following libraries:

    C++ Armadillo (http://arma.sourceforge.net/)
    llapack
    lopenblas
    C++ BOOST




COMPILE
=======

To compile using the GCC compiler, use the following commands:


    g++   -O3 -fopenmp -std=c++11   -fopenmp  code.cpp  -o  code
    -llapack -lopenblas -lgomp -lpthread -larmadillo


Make sure to replace "code.cpp" and "code" with the desired source code file. For example, to compile "mod_counter_fact.cpp", run this command:


    g++  -O3 -fopenmp -std=c++11  mod_counter_fact.cpp  -o 
    mod_counter_fact   -fopenmp  -llapack -lopenblas -lgomp -lpthread 
    -larmadillo



Execute
=======

To run the executable file (after compilation) you will need to place it in the "executables" directory. This ensures that relative file paths for reading and writing data are preserved.  Inputs are read from the "data" directory and outputs are written to the "results" directory. 


Notes
=====

This code has been compiled and run on a Linux machine. I have not tested it on a Mac or Windows machine. If you have problems, I'm happy to troubleshoot.


 





