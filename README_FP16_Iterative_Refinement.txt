Magma is releasing the Nvidia Tensor Cores version of its linear mixed-precision solver that is able to provide an FP64 solution with up to 4X speedup.
The goal is to show that Tensor Cores are not limited to artificial intelligence and we showed that high-performance computing (HPC) applications can also harness this power.

The idea is to take advantage of the FP16 performance of the Tensor Cores on Nvidia Volta (or later) GPUs, and to develop mixed-precision algorithms that use the Tensor Cores hardware and can provide solution to the FP64 accuracy by using iterative refinement techniques and mixed-precision factorizations.
That's what we call FP16-TC iterative refinement linear solver to solve Ax=b.

The API for the solver is designed to make it simple and straightforward to plug and replace LAPACK or Magma call by the new Magma API.
We encourage the scientific communities and scientific application to use this solver in particular if they deal with large matrices and 4X speedup mean something for them.
Please provide us with feedback and suggestions.


More details about the Magma FP16 Tensor cores solver for linear systems can be found in:
Azzam Haidar, Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham. 2018. Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers. In Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC '18). IEEE Press, Piscataway, NJ, USA, Article 47, 11 pages.


Iterative refinement has been known longtime ago for FP32 to FP64 where the most expensive operations are performed in FP32 and then iterative refinement using fixed point iteration is used to achieve the FP64 accuracy.
Magma releases its solver that can go from FP16 or FP32 to FP64 using different techniques for iterative refinement such as GMRES or classical iterative refinement. This provides a large set of implementations that can be studied and investigated by the research community. All versions are embedded and accessible through one expert API call where the user can specify which mixed-precision algorithm and which iterative refinement techniques must be used.
The solver has fallback to a FP64 computation in case any of the internal computations fail.

For simplicity, we also provide 2 other simple APIs that are similar to the LAPACK dsgesv API for users who want plug and replace APIs in their code without worrying about how to call the expert API.
1- The FP32 to FP64 API magma_dsgesv_iteref_gpu, which is similar to the LAPACK dsgesv API. Here A, X, and B are FP64, the routine does the internal conversion and computation, and provides FP64 solution.

2- The FP16 to FP64 API magma_dhgesv_iteref_gpu, which is similar to the magma_dsgesv_gpu API, except it does use the tensor cores and performs computations in FP16. Here A, X, and B are FP64, the routine does the internal conversion and computation, and provides FP64 solution.


Other released routines  include the Mixed precision LU factorization routines:
magma_hgetrf_gpu  (performs LU mixed precision factorization in FP16)
magma_htgetrf_gpu (performs LU mixed precision factorization in FP16-TC using Tensor Cores)


Tester:
We provided a tester (testing_dxgesv_gpu ) for the new functionality. Since the API is very similar to the existing magma_dsgesv_gpu API, the new functionality can also be called from the existing tester (testing_dsgesv_gpu) with the '--version 2’ argument (for magma_dsgesv_iteref_gpu FP32) and the '--version 3’ argument (for magma_dhegsv_iteref_gpu FP16-TC).

The tester routine can be called this way:
using the FP16-TC
numactl --interleave=all ./testing_dxgesv_gpu -N 30000 --matrix poev_arith --cond 100
numactl --interleave=all ./testing_dxgesv_gpu -N 30000 --matrix poev_arith --cond 100 --version 3
numactl --interleave=all ./testing_dsgesv_gpu -N 30000 --matrix poev_arith --cond 100 --version 3

using the FP32
numactl --interleave=all ./testing_dxgesv_gpu -N 30000 --matrix poev_arith --cond 100 --version 2
numactl --interleave=all ./testing_dsgesv_gpu -N 30000 --matrix poev_arith --cond 100 --version 2
