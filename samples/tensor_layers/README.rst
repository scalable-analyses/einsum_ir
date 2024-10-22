Tensor Layers
=====================

Tensor Ring Nets
----------------
Tensor ring net (TRN) compression of a fully connected layer in https://doi.org/10.1109/CVPR.2018.00972 for the LeNet-300-100 network with R=50.
The script ``trn.py`` was used to compute the contraction path.
The config ``trn.cfg`` for performance benchmarking was derived from the obtained output.

.. code-block:: bash

   python trn.py 2>&1 | tee trn.log