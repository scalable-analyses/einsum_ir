einsum_ir
=========

Building the Code
-----------------
.. code-block:: bash

   git clone https://github.com/libxsmm/libxsmm.git
   cd libxsmm
   git checkout feature_packed_gemm
   make BLAS=0 -j8
   cd ..

   wget https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp

   scons libtorch=/home/alex/.conda/envs/pytorch/lib/python3.9/site-packages/torch libxsmm=$(pwd)/libxsmm -j4

Benchmarking Einsum Expressions
-------------------------------
.. code-block:: bash

   ./build/bench_expression "iae,bf,dcba,cg,dh->hgfei" "32,8,4,2,16,64,8,8,8" "(1,2),(2,3),(0,1),(0,1)"