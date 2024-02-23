einsum_ir
=========

Building the Code
-----------------
.. code-block:: bash

   git clone https://github.com/scalable-analyses/einsum_ir
   cd einsum_ir

   git clone https://github.com/libxsmm/libxsmm.git
   cd libxsmm
   make BLAS=0 -j8
   cd ..

   wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.26.tar.gz
   tar -xvf v0.3.26.tar.gz
   cd OpenBLAS-0.3.26
   make -j8
   make PREFIX=$(pwd)/../openblas install
   cd ..

   wget https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp

   scons libtorch=/home/alex/.conda/envs/pytorch/lib/python3.11/site-packages/torch libxsmm=$(pwd)/libxsmm blas=$(pwd)/openblas -j4

Benchmarking Einsum Expressions
-------------------------------
.. code-block:: bash

   ./build/bench_expression "iae,bf,dcba,cg,dh->hgfei" "32,8,4,2,16,64,8,8,8" "(1,2),(2,3),(0,1),(0,1)"