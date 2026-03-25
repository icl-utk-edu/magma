from setuptools import find_packages, setup
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
import setuptools.command.install

import re
import glob
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT_DIR = os.path.abspath('.')
PACKAGE_NAME = "magma"
MAGMA_INCLUDE = os.path.join(ROOT_DIR, "include")
MAGMA_LIB = os.path.join(ROOT_DIR, "lib")
MAGMA_VERSION = "2.10.0a0"

def detect_gpu_arch():
    if os.getenv("ROCM_PATH") != "" and os.path.exists(os.path.join(os.getenv("ROCM_PATH"), "bin", "hipcc")):
        return "rocm"
    elif os.getenv("CUDA_HOME") != "" and os.path.exists(os.path.join(os.getenv("CUDA_HOME"), "bin", "nvcc")):
        return "cuda"
    else:
        print("No CUDA or ROCm installation found. Building for CPU.")
        return None


def find_library(header):

    searching_for = f"Searching for {header}"

    for folder in MAGMA_INCLUDE:
        if (Path(folder) / header).exists():
            print(f"{searching_for} in {Path(folder) / header}. Found in Magma Include.")
            return True, None, None
        
    print(f"{searching_for}. Didn't find in Magma Include.")

def get_version(arch):
    version = MAGMA_VERSION
        
    try:
        # Get version using current branch name. Will only work when building from release branch.
        branch_name = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()

        # Regular expression pattern to match 'vX.Y.Z'
        pattern = r'v(\d+\.\d+\.\d+)'
        match = re.search(pattern, branch_name)

        version = match.group(1) if match else MAGMA_VERSION # Sets version as 'X.Y.Z'

    except Exception:
        print("Could not find version from branch name.")
        pass

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        print("Could not find git SHA from branch name.")
        pass

    if arch=='rocm':
        try:
        
            with open(os.path.join(os.getenv("ROCM_PATH"), ".info", "version")) as file:
                rocm_version = file.readline().strip() 
            
            pattern = r'(\d+\.\d+\.\d+)' # Sets version as 'X.Y.Z' from something like X.Y.Z-AB
            match = re.search(pattern, rocm_version)

            version += f"+rocm{match.group(1)}" if match else ""  
        except Exception:
            print("Could not find rocm version from rocm installation.")
            pass
    print(version)

    return version, sha



def subprocess_run(command, cwd=None, env=None):
    """ This attempts to run a shell command. """
    print(f"Running: {command}")

    try:

        result = subprocess.check_output(
            command,
            cwd=ROOT_DIR,
            shell=True
        )
        return result
            
    except subprocess.CalledProcessError as e:
        print(f"Command Failed with exit code {e.returncode}:")
        print(e.stderr)
        raise
    
class Build_CMake(setuptools.command.build_py.build_py):
 
        
    def run(self):
        print("Began run function")

        self.gpu_arch = detect_gpu_arch()

        if self.gpu_arch == "cuda":
            print("Building MAGMA for CUDA...")

        elif self.gpu_arch == "rocm":

            cpus = str(int(os.cpu_count()))

            MKLROOT = "/opt/intel"
            os.environ["MKLROOT"] = MKLROOT
            os.environ["LANG"] = "C.UTF-8"
            # find theRock path
            '''
            path = subprocess.check_output(
                "find / -name libhipblas.* 2>/dev/null | head -n 1",
                shell=True,
                text=True
                ).strip()
            if not path:
                raise RuntimeError("librocblas not found (theRock not installed)")
            rocm_path = os.path.dirname(os.path.dirname(path))
            os.environ["ROCM_PATH"] = rocm_path
            os.environ["PATH"] = os.environ["PATH"] + ":" + os.path.join(rocm_path, "bin")
            '''

            print("Building MAGMA for ROCm...")

            make_inc = os.path.join(ROOT_DIR,"make.inc")

            if os.path.exists(make_inc):
                os.remove(make_inc)

            # Copy the make.inc file to this directory
            shutil.copy2(os.path.join(ROOT_DIR,"make.inc-examples","make.inc.hip-gcc-mkl"), make_inc)

            gfx_arch = os.environ.get("PYTORCH_ROCM_ARCH").split(';')

            if len(gfx_arch)==0:
                gfx_arch = subprocess.check_output(['bash', '-c', 'rocm_agent_enumerator']).decode('utf-8').split('\n') 
                
                # Remove empty entries and and gfx000
                gfx_arch = [gfx for gfx in gfx_arch if gfx not in ['gfx000', '']]


            # Writing gfx arch to file
            with open(os.path.join(make_inc), "a") as f:
                for gfx in gfx_arch:
                    f.write(f"\nDEVCCFLAGS += --offload-arch={gfx}")
                    
            # Build commands
            hip_build = f"/usr/bin/make     -f make.gen.hipMAGMA    -j {cpus}"
            so_build =  f"/usr/bin/make     lib/libmagma.so         -j {cpus}       MKLROOT={MKLROOT}"
            
            try:
                print(subprocess_run(hip_build, cwd=ROOT_DIR))
                print("End of hip build")

            except Exception as e:
                raise RuntimeError(f"Error running MAGMA library build comand: {e}") from e

            os.environ["LANG"] = 'C.UTF-8'
            try:
                print(subprocess_run(so_build, cwd=ROOT_DIR))
                print("End of main build")
            except Exception as e:
                raise RuntimeError(f"Error running MAGMA library build comand: {e}") from e
            

        build_lib = self.build_lib

        print(f"Build destination = {build_lib}")
        
        # Magma folder in wheel file
        package_target = os.path.join(ROOT_DIR, build_lib, PACKAGE_NAME)

        # Target destination for libmagma.so
        target_lib = os.path.join(package_target, "lib")

        os.makedirs(target_lib, exist_ok=True)

        # Move libmagma.so to package dir
        shutil.copy(os.path.join(MAGMA_LIB, "libmagma.so"), os.path.join(target_lib, "libmagma.so"))
        shutil.copytree(MAGMA_INCLUDE, os.path.join(package_target, "include"))

        # Call parent
        super().run()


class clean(setuptools.Command):
    
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re

        shutil.rmtree(os.path.join(ROOT_DIR, "magma.egg-info"))  if os.path.exists(os.path.join(ROOT_DIR, "magma.egg-info")) else 0
        shutil.rmtree(os.path.join(ROOT_DIR, "build")) if os.path.exists(os.path.join(ROOT_DIR, "build")) else 0
        shutil.rmtree(os.path.join(ROOT_DIR, "dist")) if os.path.exists(os.path.join(ROOT_DIR, "dist")) else 0
        os.remove(os.path.join(ROOT_DIR, "make.inc")) if os.path.exists(os.path.join(ROOT_DIR, "make.inc")) else 0

        # On ROCm install we do not 'unhipify' files when cleaning. This seems to be robust anyway.


if __name__ == "__main__":

    arch = detect_gpu_arch()
    version, sha = get_version(arch)
    version = f"{version}.dev0+g{sha[:7]}"
    version = version.replace("+g", ".g", 1)

    print(f"Building wheel {PACKAGE_NAME}-{version}")

    with open("README") as f:
        readme = f.read()

    cmdclass = {
                    "build_py": Build_CMake,
                    "clean": clean,
                }

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="ICL ",
        url="https://github.com/icl-utk-edu/magma/tree/master",
        description="",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="BSD-3-Clause",
        packages=find_packages(),
        package_data={"magma": ["lib/libmagma.so", "include/*.h"]},
        package_dir={'': '.'}, 
        include_package_data=True,
        python_requires=">=3.9",
        cmdclass=cmdclass,
    )
