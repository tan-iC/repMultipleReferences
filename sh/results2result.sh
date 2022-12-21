
###
# cd repMultipleReferences
# ./sh/results2result.sh
###

common="diffMax/parameterTuning02"
srcPath="results/${common}"
dstPath="result/${common}"

# alpha: hyper-parameter
alphas=(
    0.0
    0.5
    1.0
    5.0
    7.5
    10.0
    12.5
)

for alpha in "${alphas[@]}" ; do

    currentSrcPath="${srcPath}/${alpha}/test"
    currentDstPath="${dstPath}/${alpha}/"

    cp -r ${currentSrcPath} ${currentDstPath}

done
