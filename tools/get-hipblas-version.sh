#!/bin/bash
# get-hipblas-version.sh
#   based off get-rocm-version.sh

DEBUG=0
[[ -d $1 ]] && ROCM_PATH=$1 || ROCM_PATH="${ROCM_PATH:-/opt/ROCM_PATH}"
HEADER_PATH_ONE=${ROCM_PATH}/include/hipblas/hipblas-version.h 

function parse_semver() {
    local token="$1"
    local major=$(echo "$token" | egrep 'hipblasVersionMajor' | awk '{print $NF}')
    local minor=$(echo "$token" | egrep 'hipblasVersionMinor' | awk '{print $NF}')
    local patch=$(echo "$token" | egrep 'hipblasVersionPatch' | awk '{print $NF}')
    
    if [ -z "$major" ] || [ -z "$minor" ] || [ -z "$patch" ]
    then
        echo    "HIPBLAS Version not found inside HIPBLAS headers."
        exit 1
    fi

    echo "$major $minor $patch"
}

if [[ -f ${HEADER_PATH_ONE} ]]
then
    HIPBLAS_VERSION_DEV_RAW=$(cat ${HEADER_PATH_ONE})
    if [[ "x$DEBUG" = "x1" ]]
    then
        echo "Found ${HEADER_PATH_ONE}, HIPBLAS_VERSION_DEV_RAW=${HIPBLAS_VERSION_DEV_RAW}"
    fi
else
    echo "HIPBLAS Version not found. Please ensure ROCM_PATH is configured correctly."
    exit 1
fi

a=($(parse_semver "${HIPBLAS_VERSION_DEV_RAW}"))
major=${a[0]}
minor=${a[1]}
patch=${a[2]}

echo "$((major * 10000 + minor * 100 + patch))"

