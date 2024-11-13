
set -e # Exit on first error
# Shows the executed command but doesn't expand verbose shell functions
trap 'echo "# $BASH_COMMAND"' DEBUG # Show commands

source /apps/spacks/current/github_env/share/spack/setup-env.sh
spack env activate magma

spack load cmake
spack load python
spack load gcc

if [ "$BLAS" = "openblas" ]; then
   spack load openblas
   export OPENBLASDIR=`spack location -i openblas`
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENBLASDIR/lib
else
   spack load intel-oneapi-mkl
fi

if [ "$COMPILER" = "intel" ]; then
   spack load intel-oneapi-compilers
fi


if [ "$DEVICE" = "gpu_nvidia" ]; then
   spack load cuda
   export CUDADIR=$CUDA_HOME
   export LIBRARY_PATH=$LIBRARY_PATH:$CUDADIR/lib64
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDADIR/lib64
   export GPU_TARGET=Volta
   export BACKEND=cuda
else # DEVICE == gpu_amd
   export GPU_TARGET=gfx906
   export BACKEND=hip
   export OPTIONS="-DCMAKE_CXX_COMPILER=hipcc"
   export ROCM=/opt/rocm
   export PATH=$PATH:$ROCM/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM/lib
fi

# cd into build directory if it exists (for cmake builds)
if [ -d build ]; then
   cd build
fi
