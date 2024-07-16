#!/bin/bash

DEBUG=0
[[ -d $1 ]] && ROCM_PATH=$1 || ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HEADER_PATH_ONE=${ROCM_PATH}/include/rocm-core/rocm_version.h
HEADER_PATH_TWO=${ROCM_PATH}/include/rocm_version.h
PACKAGE_PATH_ONE=${ROCM_PATH}/.info/version-dev
PACKAGE_PATH_TWO=${ROCM_PATH}/.info/version

function parse_semver() {
    local token="$1"
    local major=0
    local minor=0
    local patch=0

    if egrep '^[0-9]+\.[0-9]+\.[0-9]+' <<<"$token" >/dev/null 2>&1
    then
        # It has the correct syntax.
        local n=${token//[!0-9]/ }
        local a=(${n//\./ })
        major=${a[0]}
        minor=${a[1]}
        patch=${a[2]}
    fi

    echo "$major $minor $patch"
}

if [[ -f ${HEADER_PATH_ONE} ]]
then
    ROCM_VERSION_DEV_RAW=$(grep ROCM_BUILD_INFO ${HEADER_PATH_ONE} | cut -d '"' -f 2)
    if [[ "x$DEBUG" = "x1" ]]
    then
        echo "Found ${HEADER_PATH_ONE}, ROCM_VERSION_DEV_RAW=${ROCM_VERSION_DEV_RAW}"
    fi
elif [[ -f ${HEADER_PATH_TWO} ]]
then
    ROCM_VERSION_DEV_RAW=$(grep ROCM_BUILD_INFO ${HEADER_PATH_TWO} | cut -d '"' -f 2)
    if [[ "x$DEBUG" = "x1" ]]
    then
        echo "Found ${HEADER_PATH_TWO}, ROCM_VERSION_DEV_RAW=${ROCM_VERSION_DEV_RAW}"
    fi
elif [[ -f ${PACKAGE_PATH_ONE} ]]
then
    ROCM_VERSION_DEV_RAW=$(cat ${PACKAGE_PATH_ONE})
    if [[ "x$DEBUG" = "x1" ]]
    then
        echo "Found ${PACKAGE_PATH_ONE}, ROCM_VERSION_DEV_RAW=${ROCM_VERSION_DEV_RAW}"
    fi
elif [[ -f ${PACKAGE_PATH_TWO} ]]
then
    ROCM_VERSION_DEV_RAW=$(cat ${PACKAGE_PATH_TWO})
    if [[ "x$DEBUG" = "x1" ]]
    then
        echo "Found ${PACKAGE_PATH_TWO}, ROCM_VERSION_DEV_RAW=${ROCM_VERSION_DEV_RAW}"
    fi
else
    echo "ROCM_VERSION_NOT_FOUND"
    exit 1
fi

a=($(parse_semver "${ROCM_VERSION_DEV_RAW}"))
major=${a[0]}
minor=${a[1]}
patch=${a[2]}

echo "$((major * 10000 + minor * 100 + patch))"

