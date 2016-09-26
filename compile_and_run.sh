mkdir build;
cd build;
rm -rf *;
cmake ../;
make;
cd ../;
export DYLD_LIBRARY_PATH=/Users/olkido/Dropbox/Work/code/other/pardiso/
export OMP_NUM_THREADS=1
./build/pardiso_example
