Tensor Decompositions
=====================

Tensor Train Decomposition
--------------------------
Tensor train (TT) decomposition of the COIL-100 dataset.
The script ``tt.py`` was used to generate the decomposition.
The config ``tt.cfg`` for performance benchmarking was derived from the obtained output.

.. code-block:: bash

   python tt.py 2>&1 | tee tt.log

Fully Connected Tensor Network
------------------------------
Fully connected tensor network (FCTN) decomposition in https://doi.org/10.1609/aaai.v35i12.17321 for the hyperspectral video (HSV) with all ranks set to 8.
The script ``fctn.py`` was used to compute the contraction path.
The config ``fctn.cfg`` for performance benchmarking was derived from the obtained output.

.. code-block:: bash

   python fctn.py 2>&1 | tee fctn.log

Tensor Wheel Decomposition
--------------------------
Tensor wheel decomposition in https://dl.acm.org/doi/10.5555/3600270.3602228 for the hyperspectral video (HSV) with R1 = R2 = R3 = R4 = 6 and  L1 = L2 = L3 = L4 = 4.
The script ``tw.py`` was used to compute the contraction path.
The config ``tw.cfg`` for performance benchmarking was derived from the obtained output.

.. code-block:: bash

   python tw.py 2>&1 | tee tw.log

Generalized model based on Tucker decomposition and Tensor Ring decomposition
-----------------------------------------------------------------------------
Tensor decomposition in https://dl.acm.org/doi/10.1145/3366423.3380188 for the n-ary relational dataset JF17K-4.
The script ``getd.py`` was used to compute the contraction path.
The config ``getd.cfg`` for performance benchmarking was derived from the obtained output.

.. code-block:: bash

   python getd.py 2>&1 | tee getd.log