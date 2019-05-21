#!/bin/sh
#
# Compile MAGMA with given build configurations,
# save output to builds/<config>/*.txt, and
# save (hardlink) libs and executables to builds/<config>/{lib, testing, sparse-testing},
# save objects to builds/<config>/obj.tar.gz.
# Takes one or more build configurations, which are suffices on make.inc.* files
# in the make.inc-examples directory.
# Usage:
#     ./tools/build.sh [-h|--help] [suffices from make.inc.*]
#
# @author Mark Gates

set -u

j=8
clean=1
sparse=1
link=""
tar=""
pause=""

usage="Usage: $0 [options] [acml macos mkl-gcc openblas ...]
    -h  --help      help
    -j #            parallel make threads, default $j
        --no-clean  skip 'make clean' before build; only one configuration allowed
        --no-sparse skip making sparse
    -l  --link      link object files
    -t  --tar       tar object files in builds/<config>/obj.tar.gz
    -p  --pause     pause after each build"


# ----------------------------------------
# parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            echo "$usage"
            exit
            ;;
        --no-clean)
            clean=""
            ;;
        --no-sparse)
            sparse=""
            ;;
        -l|--link)
            link=1
            ;;
        -t|--tar)
            tar=1
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

if [ $# -gt 1 -a -z "$clean" ]; then
    echo "building multiple configurations requires cleaning; remove --no-clean"
    exit
fi


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
    if [ ! -e make.inc-examples/make.inc.$config ]; then
        echo "Error: make.inc-examples/make.inc.$config does not exist"
        continue
    fi
    rm make.inc
    ln -s  make.inc-examples/make.inc.$config  make.inc

    if [ -d $builds/$config ]; then
        echo "$builds/$config already exists; creating new directory"
        count=2
        while [ -d $builds/$config-$count ]; do
            count=$((count+1))
        done
        config=$config-$count
    fi
    echo "building in $builds/$config"
    mkdir $builds/$config

    if [ -n "$clean" ]; then
        echo "$make clean"
        touch make.inc
        $make clean > /dev/null
    else
        echo "SKIPPING CLEAN"
    fi

    run "$make lib"             $builds/$config/out-lib.txt          $builds/$config/err-lib.txt
    run "$make test -k"         $builds/$config/out-test.txt         $builds/$config/err-test.txt
    if [ -n "$sparse" ]; then
        run "$make sparse-lib"      $builds/$config/out-sparse-lib.txt   $builds/$config/err-sparse-lib.txt
        run "$make sparse-test -k"  $builds/$config/out-sparse-test.txt  $builds/$config/err-sparse-test.txt
    else
        echo "SKIPPING SPARSE"
    fi

    mkdir $builds/$config/lib
    mkdir $builds/$config/testing

    mkdir $builds/$config/sparse
    mkdir $builds/$config/sparse/testing

    # link libraries
    ln lib/lib*       $builds/$config/lib

    # link testers
    ln `find testing        -maxdepth 1 -perm -u+x -type f -not -name '*.py' -not -name '*.pl' -not -name '*.sh' -not -name '*.csh'` $builds/$config/testing
    (cd $builds/$config/testing; ln -s ../../../../testing/run* .; ln -s ../../../../testing/*.ref .)
    pwd
    if [ -n "$sparse" ]; then
        ln `find sparse/testing -maxdepth 1 -perm -u+x -type f -not -name '*.py' -not -name '*.pl' -not -name '*.sh' -not -name '*.csh'` $builds/$config/sparse/testing
    fi

    # link objects
    if [ -n "$link" ]; then
        mkdir $builds/$config/src
        mkdir $builds/$config/control
        mkdir $builds/$config/magmablas
        mkdir $builds/$config/interface_cuda

        mkdir $builds/$config/sparse/src
        mkdir $builds/$config/sparse/control
        mkdir $builds/$config/sparse/blas

        ln src/*.o            $builds/$config/src
        ln control/*.o        $builds/$config/control
        ln magmablas/*.o      $builds/$config/magmablas
        ln interface_cuda/*.o $builds/$config/interface_cuda
        ln testing/*.o        $builds/$config/testing

        ln sparse/src/*.o     $builds/$config/sparse/src
        ln sparse/control/*.o $builds/$config/sparse/control
        ln sparse/blas/*.o    $builds/$config/sparse/blas
        ln sparse/testing/*.o $builds/$config/sparse/testing
    fi

    # tar objects
    if [ -n "$tar" ]; then
        echo "tar objs " `date`
        ./tools/find_obj_files.sh > obj-files.txt
        tar -zcf $builds/$config/obj.tar.gz -T obj-files.txt
        echo "done     " `date`
    else
        echo "SKIPPING TAR"
    fi

    if [ -n "$pause" ]; then
        echo "[return to continue]"
        read
    fi
done
