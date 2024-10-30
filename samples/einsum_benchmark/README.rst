Einsum Benchmark
----------------
The script ``benchmark_convert.py`` can be used to convert benchmarks from https://benchmark.einsum.org/ to a format suitable for einsum_ir.

.. code-block:: bash

   wget https://zenodo.org/records/11477304/files/instances.zip -O instances.zip
   unzip instances.zip
   python benchmark_convert.py instances/lm_first_last_brackets_4_16d.pkl > lm_first_last_brackets_4_16d.cfg
   python benchmark_convert.py instances/lm_batch_likelihood_sentence_5_16d.pkl | sed 's/3500/128/g' > lm_batch_likelihood_sentence_5_16d_batch128.cfg
   python benchmark_convert.py instances/str_nw_mera_open_26.pkl > str_nw_mera_open_26.cfg

TVM
---
.. code-block:: bash

   today=$(date +"%y_%m_%d")
   PYTHONUNBUFFERED=1 python tvm_synthetic.py --uarch neoverse_v2 2>&1 | tee tvm_synthetic_$today.log
