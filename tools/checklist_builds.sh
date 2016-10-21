#!/bin/sh
#
# Compiles MAGMA with given build configurations,
# saving output to builds/<config>/*.txt, and
# saving objects and executables to builds/<config>/obj.tar.gz
# Takes one or more build configurations, which are suffix on make.inc.* files.
# Usage:
#     ./tools/checklist_builds.sh [-h|--help] [suffices from make.inc.*]
#
# @author Mark Gates

set -u

j=8

usage="Usage: $0 [-c] [-t] [-p] [-j #] [acml macos mkl-gcc openblas ...]
    -h  --help   help
    -c  --clean  clean
    -j #         parallel make threads, default $j
    -s  --save   save lib and executables
    -t  --tar    tar object files and executables
    -p  --pause  pause after each build"


# ----------------------------------------
# parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            echo "$usage"
            exit
            ;;
        -c|--clean)
            clean=1
            ;;
        -t|--tar)
            tar=1
            ;;
        -s|--save)
            save=1
            ;;
        -p|--pause)
            pause=1
            ;;
        -j)
            j=$2
            shift
            ;;
        --)
            shift
            break
            ;;
        -?*)
            echo "Error: unknown option: $1" >& 2
            exit
            ;;
        *)
            break
            ;;
    esac
    shift
done


# ----------------------------------------
# usage: sep filename
# appends separator to existing file
function sep {
    if [ -e $1 ]; then
        echo "####################" `date` >> $1
    fi
}


# ----------------------------------------
# usage: run command output-filename error-filename
# runs command, saving stdout and stderr, and print error if it fails
function run {
    sep $2
    sep $3
    printf "%-32s %s\n"  "$1"  "`date`"
    #echo "$1 " `date`
    $1 >> $2 2>> $3
    if (($? > 0)); then
        echo "FAILED"
    fi
}


# ----------------------------------------
builds=builds/`date +%Y-%m-%d`
echo "builds directory $builds"

make="make -j$j"

if [ ! -d builds ]; then
    mkdir builds
fi

if [ ! -d $builds ]; then
    mkdir $builds
fi

for config in $@; do
    echo "========================================"
    echo "config $config"
    echo
    if [ ! -e make.inc.$config ]; then
        echo "Error: make.inc.$config does not exist"
        if [ -n "${pause+set}" ]; then
            echo "[return to continue]"
            read
        fi
        continue
    fi
    rm make.inc
    ln -s  make.inc.$config  make.inc
    mkdir $builds/$config
    
    if [ -n "${clean+set}" ]; then
        echo "$make clean"
        touch make.inc
        $make clean > /dev/null
    else
        echo "SKIPPING CLEAN"
    fi
    
    run "$make lib"             $builds/$config/out-lib.txt          $builds/$config/err-lib.txt
    run "$make test -k"         $builds/$config/out-test.txt         $builds/$config/err-test.txt
    run "$make sparse-lib"      $builds/$config/out-sparse-lib.txt   $builds/$config/err-sparse-lib.txt
    run "$make sparse-test -k"  $builds/$config/out-sparse-test.txt  $builds/$config/err-sparse-test.txt
    
    if [ -n "${save+set}" ]; then
        echo "saving libs and executables to $builds/$config/{lib, testing, sparse-testing}"
        mkdir $builds/$config/lib
        mkdir $builds/$config/testing
        mkdir $builds/$config/sparse-testing
        mv lib/lib* $builds/$config/lib
        mv `find testing        -maxdepth 1 -perm -u+x -type f -not -name '*.py' -not -name '*.pl' -not -name '*.sh' -not -name '*.csh'` $builds/$config/testing
        mv `find sparse/testing -maxdepth 1 -perm -u+x -type f -not -name '*.py' -not -name '*.pl' -not -name '*.sh' -not -name '*.csh'` $builds/$config/sparse-testing
    else
        echo "SKIPPING SAVE"
    fi
    
    if [ -n "${tar+set}" ]; then
        echo "tar objs " `date`
        ./tools/find_obj_files.sh > obj-files.txt
        tar -zcf $builds/$config/obj.tar.gz -T obj-files.txt
        echo "done     " `date`
    else
        echo "SKIPPING TAR"
    fi
    
    if [ -n "${pause+set}" ]; then
        echo "[return to continue]"
        read
    fi
    
    if [ -z "${clean+set}" ]; then
        break
    fi
done
