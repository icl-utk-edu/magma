import ctypes
from magma_interface import *
from datetime import datetime

#libmagma = ctypes.CDLL('/home/hanzt/magma_dev/magma/trunk/lib/libmagma.so')
#libmagma_sparse = ctypes.CDLL('/home/hanzt/magma_dev/magma/trunk/lib/libmagma_sparse.so')


A = MATRIX()
B = MATRIX()
x = VECTOR()
b = VECTOR()

matrix_read( ctypes.byref(A), "../testing/test_matrices/Trefethen_20.mtx" )
matrix_transfer( A, ctypes.byref(B), CPU, DEV )


vector_init( ctypes.byref(b), DEV, A.num_rows, c_double(5.0) )
vector_init( ctypes.byref(x), DEV, A.num_rows, c_double(1.0) )
vector_visualize( x, 0, 5 )

solver_par = solver_parameter()
solver_par.maxiter=10000
solver_par.epsilon=0.0000001

#mv( c_double(1.0), B, b, c_double(0.0), x )

startTime = datetime.now()

cg( B, b, ctypes.byref(x), ctypes.byref(solver_par) )

endTime = datetime.now()

vector_visualize( x, 0, 5 )


print "iterations: %d" %solver_par.numiter
print (endTime-startTime)

matrix_free(ctypes.byref(A))
matrix_free(ctypes.byref(B))
