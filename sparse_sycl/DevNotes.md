# Issues  
-  No `cusparse` equivalent calls work yet which affects to all routines in the following files:
+ `zmergecg.dp.cpp`
+ `magma_trisolve.cpp`
+ `magma_zcuspmm.cpp`
+ `magma_zcuspaxpy.cpp`
+ `magma_zmatrixtools_gpu.dp.cpp`


- For now, all `tex1Dfetch` `tex2Dfetch` ... do not work, a temp solution (for compilation) is commenting out all calls to those functions, thus `zmgesellcmmv` does not work.

- `dmergecg.dp.cpp` includes `<real>` which does not exist, the same with `magma_dmatrixtools_gpu.dp.cpp`.  

- `MAGMA_SIVERGENCE` issue with `codegen` in scg.cpp. Temp solution: defining `MAGMA_SIVERGENCE = MAGMA_DIVERGENCE` in the `include/magmasparse.h`.
