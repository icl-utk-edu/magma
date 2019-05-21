#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $version;
my $major;
my $minor;
my $micro;

# default options
my $rc       = 0;  # release candidate
my $alpha    = 0;
my $beta     = 0;
my $revision = '';

# ------------------------------------------------------------------------------
# In alphabetic order
# Using qw() avoids need for "quotes", but comments aren't recognized inside qw()
my @files2delete = qw(
    ReleaseChecklist
    Makefile.internal
    control/sizeptr

    docs/Doxyfile-fast
    docs/output_err

    include/Makefile
    make.inc
    make.inc-examples/make.inc.ig.pgi
    scripts

    sparse/python

    sparse/testing/test_matrices
    sparse/testing/testing_zpardiso.cpp
    sparse/testing/testing_zparilu_weight.cpp
    sparse/testing/testing_zsolver_allufmc.cpp
    sparse/testing/testing_zsolver_energy.cpp

    sparse/blas/zilut.cpp

    testing/*.txt
    testing/fortran2.cpp
    testing/testing_zgetrf_f.f
    testing/testing_zgetrf_gpu_f.cuf

    tools/MakeMagmaRelease.pl
    tools/checklist.csh
    tools/checklist_ceildiv.pl
    tools/checklist_run_tests.pl
    tools/compare_prototypes.pl
    tools/fortran_wrappers.pl
    tools/magmasubs.pyc
    tools/parse-magma.py
    tools/trim_spaces.pl
    tools/wdiff.pl
);
# note: keep tools/{codegen.py, magmasubs.py}


# ------------------------------------------------------------------------------
sub myCmd
{
    my ($cmd) = @_;
    print '-' x 80, "\n";
    print $cmd."\n";
    my $err = system($cmd);
    if ($err != 0) {
        print "Error during execution of the following command:\n$cmd\n";
        exit;
    }
}


# ------------------------------------------------------------------------------
sub MakeRelease
{
    my $cmd;
    my $stage = "";

    if ( $rc > 0 ) {
        $version .= "-rc$rc";
        $stage = "rc$rc";
    }
    if ( $alpha > 0 ) {
        $version .= "-alpha$alpha";
        $stage = "alpha$alpha";
    }
    if ( $beta > 0 ) {
        $version .= "-beta$beta";
        $stage = "beta$beta";
    }
    my($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime;
    my @months = (
        'January',   'February', 'March',    'April',
        'May',       'June',     'July',     'August',
        'September', 'October',  'November', 'December',
    );
    $year += 1900;

    # Require recent doxygen, say >= 1.8.
    # ICL machines have ancient versions of doxygen (1.4 and 1.6);
    # the docs don't work at all.
    my $doxygen = `doxygen --version`;
    chomp $doxygen;
    my($v) = $doxygen =~ m/^(\d+\.\d+)\.\d+$/;
    my $doxygen_require = 1.8;
    if ( $v < $doxygen_require ) {
        print <<EOT;
=====================================================================
WARNING: MAGMA requires doxygen version $doxygen_require; installed version is $doxygen.
The documentation will not be generated correctly.
Look in /mnt/sw/doxygen for a recent release.
Enter 'ignore' to acknowledge, or control-C to go fix the problem.
=====================================================================
EOT
        do {
            $_ = <STDIN>;
        } while( not m/\bignore\b/ );
    }

    my $RELEASE_PATH = $ENV{PWD}."/magma-$version";
    if ( -e $RELEASE_PATH ) {
        die( "RELEASE_PATH $RELEASE_PATH already exists.\nPlease delete it or use different version.\n" );
    }

    # Save current directory
    my $dir = `pwd`;
    chomp $dir;

    if ( not $rc and not $alpha and not $beta ) {
        print "Update MAGMA version in include headers and tag release (y/n)? ";
        $_ = <STDIN>;
        if ( m/\b(y|yes)\b/ ) {
            # If run as ./tools/MakeMagmaRelease.pl, no need to change dir;
            # if run as ./MakeMagmaRelease.pl, cd up to trunk, adjust includes, then cd back.
            if ( $dir =~ m|/tools$| ) {
                chdir "..";
            }
            $cmd = "perl -pi -e 's/VERSION_MAJOR +[0-9]+/VERSION_MAJOR $major/; "
                 .             " s/VERSION_MINOR +[0-9]+/VERSION_MINOR $minor/; "
                 .             " s/VERSION_MICRO +[0-9]+/VERSION_MICRO $micro/;' "
                 .             " include/magma_types.h";
            myCmd($cmd);
            myCmd("perl -pi -e 's/PROJECT_NUMBER +=.*/PROJECT_NUMBER         = $major.$minor.$micro/' docs/Doxyfile");
            myCmd("perl -pi -e 's/(Copyright 2009)-\\d{4}/\$1-$year/' COPYRIGHT");
            myCmd("hg diff include/magma_types.h docs/Doxyfile COPYRIGHT");
            print "Commit these changes now (y/n)? ";
            $_ = <STDIN>;
            if ( m/\b(y|yes)\b/ ) {
                myCmd("hg commit -m 'version $major.$minor.$micro' include/magma_types.h docs/Doxyfile COPYRIGHT");
            }

            print "Tag release in Mercurial (y/n)? ";
            $_ = <STDIN>;
            if ( m/\b(y|yes)\b/ ) {
                myCmd("hg tag v$version");
            }

            myCmd("hg out");
            print "Push the above changes now (y/n)? ";
            $_ = <STDIN>;
            if ( m/\b(y|yes)\b/ ) {
                myCmd("hg push");
            }
            else {
                print "Remember to push changes later!\n";
            }

            # allow user to see that it went successfully
            print "Type enter to continue\n";
            $_ = <STDIN>;

            chdir $dir;
        }
    }

    $cmd = "hg archive $revision $RELEASE_PATH";
    myCmd($cmd);

    print "cd $RELEASE_PATH\n";
    chdir $RELEASE_PATH;

    # Change version in magma.h (in export, not in the hg repo itself)
    # See similar replacement above (minus stage)
    $cmd = "perl -pi -e 's/VERSION_MAJOR +[0-9]+/VERSION_MAJOR $major/; "
         .             " s/VERSION_MINOR +[0-9]+/VERSION_MINOR $minor/; "
         .             " s/VERSION_MICRO +[0-9]+/VERSION_MICRO $micro/; "
         .             " s/VERSION_STAGE +.+/VERSION_STAGE \"$stage\"/;' "
         .             " include/magma_types.h";
    myCmd($cmd);
    myCmd("perl -pi -e 's/PROJECT_NUMBER +=.*/PROJECT_NUMBER         = $major.$minor.$micro/' docs/Doxyfile");

    # Change the version and date in comments
    my $date = "$months[$mon] $year";
    my $script = "s/MAGMA \\\(version [0-9.]+\\\)/MAGMA (version $version)/;";
    $script .= " s/\\\@date.*/\\\@date $date/;";
    myCmd("find . -type f -exec perl -pi -e '$script' {} \\;");

    # Change version in pkgconfig
    $script = "s/Version: [0-9.]+/Version: $version/;";
    myCmd("perl -pi -e '$script' lib/pkgconfig/magma.pc.in");

    # Precision Generation
    # Need to define Fortran compiler to something that exists (true),
    # in order to include all Fortran files in CMake.src.
    print "Generate the different precisions\n";
    myCmd("echo 'FORT = true' > make.inc");
    myCmd("make -j generate");

    # Compile the documentation
    print "Compile the documentation\n";
    myCmd("make -C ./docs");
    myCmd("rm -f make.inc");

    # Remove non-required files (e.g., Makefile.gen)
    myCmd("rm -rf @files2delete");

    # Remove the lines relative to include directory in root Makefile
    myCmd("perl -ni -e 'print unless /cd include/' $RELEASE_PATH/Makefile");

    # Remove '.Makefile.gen files'
    myCmd("find $RELEASE_PATH -name .Makefile.gen -exec rm -f {} \\;");

    chdir $dir;

    # Save the InstallationGuide if we want to do a plasma-installer release
    #myCmd("cp $RELEASE_PATH/InstallationGuide README-installer");

    # Create tarball
    print "Create the tarball\n";
    my $DIRNAME  = `dirname $RELEASE_PATH`;
    my $BASENAME = `basename $RELEASE_PATH`;
    chomp $DIRNAME;
    chomp $BASENAME;
    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
}


# ------------------------------------------------------------------------------
sub Usage
{
    print "MakeRelease.pl [options] major.minor.micro\n";
    print "   -h            Print this help\n";
    print "   -a alpha      Alpha version\n";
    print "   -b beta       Beta version\n";
    print "   -c candidate  Release candidate number\n";
    print "   -r revision   Choose hg revision number\n";
}


# ------------------------------------------------------------------------------
my %opts;
getopts("ha:b:c:r:",\%opts);

if ( defined $opts{h}  ) {
    Usage();
    exit;
}
if ( defined $opts{a} ) {
    $alpha = $opts{a};
}
if ( defined $opts{b} ) {
    $beta = $opts{b};
}
if ( defined $opts{c} ) {
    $rc = $opts{c};
}
if ( defined $opts{r} ) {
    $revision = "-r $opts{r}";
}
if ( ($#ARGV + 1) != 1 ) {
    Usage();
    exit;
}

$version = shift;
($major, $minor, $micro) = $version =~ m/^(\d+)\.(\d+)\.(\d+)$/;
if ( not ($major >= 1 and $minor >= 0 and $micro >= 0)) {
    Usage();
    exit;
}

MakeRelease();
