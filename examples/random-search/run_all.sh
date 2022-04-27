BASE=`pwd`
for d in muon-airss-out/dftb+/*
do
	echo "Running " $d
	cd $d
	dftb+ > dftb.out
	cd $BASE
done
