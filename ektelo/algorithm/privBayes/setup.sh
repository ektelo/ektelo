rm *.so
rm privBayes.cpp
rm -r build

python setup.py build_ext --inplace 
