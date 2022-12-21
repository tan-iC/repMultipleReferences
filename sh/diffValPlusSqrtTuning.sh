###
### fine-tuning (ディクショナリ統一)
###

###
# cd repMultipleReferences
# ./sh/diffValPlusSqrtTuning.sh
###

# method
method="diffValPlusSqrt"

# dir_name
setting="parameterTuning01"

# feature
feature="diff, dropout=0.0, valid=sentence_pair, vocab_size=32,000"

# patience
patience=5

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

# dst dir (preprocess)
BIN_DATA="data/binarized/finetune/finetune32k"

###
# execute
###
cp_path="data/checkpoints"
basepath="${cp_path}/${method}"
mkdir "${basepath}"

basepath="${basepath}/${setting}"
mkdir "${basepath}"

echo "$0 start `date "+%Y-%m-%d-%H-%M-%S"`" >> "${basepath}/README.txt"
echo "${basepath}" >> "${basepath}/README.txt"
echo "${setting}" >> "${basepath}/README.txt"
echo "${feature}" >> "${basepath}/README.txt"

# copy pretrain data to finetune
mkdir "${basepath}/template"
cp "data/checkpoints/pretrain/checkpoint_best.pt" "${basepath}/template/"


# preprocess
# fairseq-preprocess \
#         --user-dir scripts \
#         --source-lang src --target-lang tgt \
#         --task add_args_translation \
#         --srcdict data/binarized/common/dict.ja.txt \
#         --tgtdict data/binarized/common/dict.en.txt \
#         --trainpref data/prepared/finetune/setData/train \
#         --validpref data/prepared/finetune/dev \
#         --testpref data/prepared/finetune/test \
#         --destdir ${BIN_DATA} \
#         --bpe=sentencepiece \
#         --workers 8


for alpha in "${alphas[@]}" ; do
    echo -e "\t${alpha} start `date "+%Y-%m-%d-%H-%M-%S"`" >> "${basepath}/README.txt"

    current="${basepath}/${alpha}"
    cp -r "${basepath}/template" "${current}"

    # fine-tuning
    CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train \
        $BIN_DATA/ \
        --user-dir scripts \
        --source-lang src --target-lang tgt \
        --criterion-alpha "${alpha}" \
        --task add_args_translation \
        --log-format json --log-file "log/finetune/${method}_${setting}_${alpha}.json" \
        --finetune-from-model "${current}/checkpoint_best.pt" \
        --arch transformer --share-decoder-input-output-embed --activation-fn relu \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
        --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
        --weight-decay 0.0001 --dropout 0.0 \
        --criterion multi_ref_diff_val_plus_sqrt_loss \
        --batch-size 200 --patience ${patience} \
        --save-dir "${current}" \
        --max-epoch 100 --keep-best-checkpoints 2 \
        --bpe=sentencepiece \
        --fp16

    # generate
    fairseq-generate \
        $BIN_DATA/ \
        --user-dir scripts \
        --source-lang src --target-lang tgt \
        --task add_args_translation \
        --gen-subset test \
        --path "${current}/checkpoint_best.pt" \
        --max-len-a 1 --max-len-b 50 \
        --beam 5 --lenpen 1.0 \
        --nbest 1 \
        --remove-bpe=sentencepiece \
        --results-path "result/${method}/${setting}/${alpha}/test" \
        --fp16

    # generate
    fairseq-generate \
        $BIN_DATA/ \
        --user-dir scripts \
        --source-lang src --target-lang tgt \
        --task add_args_translation \
        --gen-subset valid \
        --path "${current}/checkpoint_best.pt" \
        --max-len-a 1 --max-len-b 50 \
        --beam 5 --lenpen 1.0 \
        --nbest 1 \
        --remove-bpe=sentencepiece \
        --results-path "result/${method}/${setting}/${alpha}/val" \
        --fp16

    echo -e "\t${alpha} done `date "+%Y-%m-%d-%H-%M-%S"`" >> "${basepath}/README.txt"

done

echo "$0 was done `date "+%Y-%m-%d-%H-%M-%S"`" >> "${basepath}/README.txt"
