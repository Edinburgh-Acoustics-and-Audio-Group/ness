#!/bin/bash
#
# Script for running all the system test cases
#

# quieten down the logs so that only important messages are seen
export NESS_LOG_LEVEL=5

failures=0

# loop over the test case directories
for testcase in testcase_* ; do

    cd $testcase
    echo "Running $testcase"

    # run the code, request raw outputs
    ../../ness-framework -r

    # ensure code finished normally
    if [ $? -eq 0 ] ; then

	# compare each output with the expected result
	for i in output*.f64 ; do
	    ../compare $i gold-$i 1e-8
	    if [ $? -eq 1 ] ; then
		echo "Output $i doesn't match expected result!"
		failures=$[$failures + 1]
	    elif [ $? -eq 2 ] ; then
		echo "Error running comparison program - did you run make in the systemtests tree?"

		# bail out now, because it will probably fail every time
		exit 1
	    fi
	done
    else
	echo "Error running code for ${testcase}!"
	failures=$[$failures + 1]
    fi

    # clean up test results
    rm -f *.wav output*.f64

    # back up to systemtest directory
    cd ..
done

if [ $failures -eq 0 ] ; then
    echo
    echo "All tests passed successfully"
else
    echo "$failures tests FAILED!"
fi
