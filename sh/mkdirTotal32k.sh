###
### mkdir for total32k
###

###
# cd repMultipleReferences
# ./sh/mkdirTotal32k.sh
###
# - repMultipleReferences
#     1. total32k
#         1. data
#             1. binarized
#                 1. common
#                 1. pretrain
#                 1. finetune

#             1. checkpoints
#                 1. pretrain
#                 1. finetune

#             1. prepared
#                 1. combined
#                 1. pretrain
#                 1. finetune
#                     1. setData

#             1. spm

#         1. result
#             1. pretrain
#             1. finetune

#         1. log


# base
base="total32k"
mkdir $base
echo "${base}"

# 
dirs=(
    "data/binarized/common"
    "data/binarized/pretrain"
    "data/binarized/finetune"
    "data/checkpoints/pretrain"
    "data/checkpoints/finetune"
    "data/preapred/combined"
    "data/preapred/pretrain"
    "data/preapred/finetune/setData"
    "data/spm"
    "result/pretrain"
    "result/finetune"
    "log"
)

for current in "${dirs[@]}" ; do
    mkdir -p "${base}/${current}"
    echo "${current}"
done
