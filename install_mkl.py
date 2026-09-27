import subprocess
import os

MKL_VERSION = "2024.2.0"
MKLROOT = os.environ.get("MKLROOT", "/opt/intel")
print(f"Installing MKL to {MKLROOT}")

os.makedirs(MKLROOT, exist_ok=True)

subprocess.check_call("python3 -m pip install -U wheel", shell=True)

subprocess.check_call(
    f"python3 -m pip download mkl-static=={MKL_VERSION} mkl-include=={MKL_VERSION}",
    shell=True,
)

subprocess.check_call(
    f"python3 -m wheel unpack mkl_static-{MKL_VERSION}-py2.py3-none-manylinux1_x86_64.whl",
    shell=True,
)

subprocess.check_call(
    f"python3 -m wheel unpack mkl_include-{MKL_VERSION}-py2.py3-none-manylinux1_x86_64.whl",
    shell=True,
)

subprocess.check_call(
    f"mv mkl_static-{MKL_VERSION}/mkl_static-{MKL_VERSION}.data/data/lib {MKLROOT}",
    shell=True,
)

subprocess.check_call(
    f"mv mkl_include-{MKL_VERSION}/mkl_include-{MKL_VERSION}.data/data/include {MKLROOT}",
    shell=True,
)

os.makedirs(f"{MKLROOT}/lib/intel64", exist_ok=True)

subprocess.check_call(
    f"ln -sf {MKLROOT}/lib/libmkl*.a {MKLROOT}/lib/intel64/",
    shell=True,
)
