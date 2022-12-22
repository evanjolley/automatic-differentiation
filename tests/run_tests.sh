#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script

set -e # Exit if command returns with non-zero status

# list of test cases you want to run
tests=(
    # test_other_things_on_root_level.py
    # subpkg_1/test_module_1.py
    # subpkg_1/test_module_2.py
    test_differentiate.py
    # test_dualnums.py
    test_node.py
    test_functions.py
)

# gets present directory, goes back, then goes into src.
export PYTHONPATH="$(pwd -P)/../src/autodiff_package":${PYTHONPATH}

if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="pytest --cov=../src/autodiff_package/. --cov-report=html:htmlcov --cov-report=xml:cov.xml"
    # driver2="pytest --cov=../autodiff_package/. | grep 'TOTAL' | awk '{print $4; }'"
    ${driver}
else
    driver="pytest"
    ${driver} ${tests[@]}
fi

# pytest --cov | grep 'TOTAL' | awk '{print $4; }')

# Only use pytest for now..

#$pytest ${tests[@]} # Not sure what needs to call in tests

