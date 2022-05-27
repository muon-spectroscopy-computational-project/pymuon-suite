BASE=`pwd`
for d in ethyleneMu_opt_displaced/*
do
	echo "Running " $d
	cd $d
	dftb+ > dftb.out
	cd $BASE
done
