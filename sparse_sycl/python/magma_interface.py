import ctypes
from ctypes import *

#libmagma = ctypes.CDLL('/home/hanzt/magma_dev/magma/trunk/lib/libmagma.so')
#libmagma_sparse = ctypes.CDLL('/home/hanzt/magma_dev/magma/trunk/lib/libmagma_sparse.so')

class MATRIX(Structure):
    _fields_ = [("storage_type", c_int),
                ("memory_location", c_int),
                ("num_rows", c_int),
                ("num_cols", c_int),
                ("nnz", c_int),
                ("max_nnz_row", c_int),
                ("diameter", c_int),
                ("val", POINTER(c_double)),
                ("row", POINTER(c_int)),
                ("col", POINTER(c_int)),
                ("blockinfo", POINTER(c_int)),
                ("blocksize", c_int),
                ("numblocks", c_int)]

class VECTOR(Structure):
    _fields_ = [("memory_location", c_int),
                ("num_rows", c_int),
                ("nnz", c_int),
                ("val", POINTER(c_double))]

class solver_parameter(Structure):
    _fields_ = [("epsilon", c_double),
                ("maxiter", c_int),
                ("restart", c_int),
                ("numiter", c_int),
                ("residual",c_double)]

class precond_parameter(Structure):
    _fields_ = [("precond_type", c_double),
                ("precision", c_int),
                ("epsilon", c_double),
                ("maxiter", c_int),
                ("restart", c_int),
                ("numiter", c_int),
                ("residual",c_double)]


# matrix functions
matrix_read=libmagma_sparse.magma_d_csr_mtx
matrix_visualize=libmagma_sparse.magma_d_mvisu
matrix_write=libmagma_sparse.write_d_csrtomtx
matrix_transpose=libmagma_sparse.magma_d_mtranspose
matrix_transfer=libmagma_sparse.magma_d_mtransfer
matrix_convert=libmagma_sparse.magma_d_mconvert
matrix_free=libmagma_sparse.magma_d_mfree


# vector functions
vector_init=libmagma_sparse.magma_d_vinit
vector_read=libmagma_sparse.magma_d_vread
vector_visualize=libmagma_sparse.magma_d_vvisu
vector_transfer=libmagma_sparse.magma_d_vtransfer
vector_free=libmagma_sparse.magma_d_vfree

#iterative linear solvers
cg=libmagma_sparse.magma_dcg
gmres=libmagma_sparse.magma_dgmres
bicgstab=libmagma_sparse.magma_dbicgstab
pcg=libmagma_sparse.magma_dpcg
pgmres=libmagma_sparse.magma_dpcg
dpbicgstab=libmagma_sparse.magma_dpcg
jacobi=libmagma_sparse.magma_djacobi
ir=libmagma_sparse.magma_dir
mv=libmagma_sparse.magma_d_spmv

CSR           = 411
ELLPACK       = 412
ELLPACKT      = 413
DENSE         = 414  
BCSR          = 415
CSC           = 416
HYB           = 417
COO           = 418

CPU           = 421
DEV           = 422

CG            = 431
GMRES         = 432
BICGSTAB      = 433
JACOBI        = 434
GS            = 435

DCOMPLEX      = 451
FCOMPLEX      = 452
DOUBLE        = 453
FLOAT         = 454
