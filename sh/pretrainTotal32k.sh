
###
# cd repMultipleReferences
# ./sh/pretrainTotal32k.sh
###

###
### pretrain
###
TEXT="total32k/data/prepared"
DICT="total32k/data/binarized/common"
COMBINED="${TEXT}/combined"
PRETRAIN="${TEXT}/pretrain"
BIN_DATA="total32k/data/binarized/pretrain"
PRETRAIN_CP="total32k/data/checkpoints/pretrain"

###
# preprocess
###
# 統合ディクショナリ作成
# fairseq-preprocess \
#        --source-lang ja --target-lang en \
#        --user-dir scripts \
#        --task add_args_translation \
#        --trainpref "${COMBINED}/train" \
#        --validpref "${PRETRAIN}/dev" \
#        --testpref "${PRETRAIN}/devtest" \
#        --destdir "${DICT}" \
#        --bpe=sentencepiece \
#        --joined-dictionary \
#        --workers 20

# pretrainデータの前処理
# fairseq-preprocess \
#        --source-lang ja --target-lang en \
#        --user-dir scripts \
#        --task add_args_translation \
#        --srcdict "${DICT}/dict.ja.txt" \
#        --tgtdict "${DICT}/dict.en.txt" \
#        --trainpref "${PRETRAIN}/train" \
#        --validpref "${PRETRAIN}/dev" \
#        --testpref "${PRETRAIN}/devtest" \
#        --destdir "${BIN_DATA}" \
#        --bpe=sentencepiece \
#        --workers 20

###
# train
###
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train \
       "${BIN_DATA}/" \
       --source-lang ja --target-lang en \
       --user-dir scripts \
       --task add_args_translation \
       --log-format json --log-file "total32k/log/pretrain.json" \
       --arch transformer --share-all-embeddings \
       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
       --dropout 0.1 --weight-decay 1e-5 \
       --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
       --max-tokens 30000 --patience 10 \
       --save-dir "${PRETRAIN_CP}" \
       --max-epoch 100 --save-interval-updates 2000 \
       --keep-last-epochs 10 --keep-best-checkpoints 2 \
       --bpe=sentencepiece \
       --fp16

# generate
# fairseq-generate \
#        "${BIN_DATA}/" \
#        --source-lang ja --target-lang en \
#        --user-dir ../scripts \
#        --task add_args_translation \
#        --gen-subset test \
#        --path data/checkpoints/pretrain/checkpoint_best.pt \
#        --max-len-a 1 --max-len-b 50 \
#        --beam 5 --lenpen 1.0 \
#        --nbest 1 \
#        --remove-bpe=sentencepiece \
#        --results-path data/result \
#        --fp16
