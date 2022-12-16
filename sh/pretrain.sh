
###
# cd repMultipleReferences
# ./sh/pretrain.sh
###

###
### pretrain
###
TEXT="data/prepared"
DICT="data/binarized/common"
COMBINED="${TEXT}/combined"
PRETRAIN="${TEXT}/pretrain"
BIN_DATA="data/binarized/pretrain"

### 統合ディクショナリ作成
fairseq-preprocess \
       --source-lang ja --target-lang en \
       --user-dir scripts \
       --task add_args_translation \
       --trainpref "${COMBINED}/train" \
       --validpref "${PRETRAIN}/dev" \
       --testpref "${PRETRAIN}/devtest" \
       --destdir "${DICT}" \
       --bpe=sentencepiece \
       --workers 8

### pretrainデータの前処理
fairseq-preprocess \
       --source-lang ja --target-lang en \
       --user-dir scripts \
       --task add_args_translation \
       --srcdict "${DICT}/dict.ja.txt" \
       --tgtdict "${DICT}/dict.en.txt" \
       --trainpref "${PRETRAIN}/train" \
       --validpref "${PRETRAIN}/dev" \
       --testpref "${PRETRAIN}/devtest" \
       --destdir "${BIN_DATA}" \
       --bpe=sentencepiece \
       --workers 8

# train
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-train \
       "${BIN_DATA}/" \
       --source-lang ja --target-lang en \
       --user-dir scripts \
       --task add_args_translation \
       --log-format json --log-file "log/pretrain/pretrain.json" \
       --arch transformer --share-decoder-input-output-embed --activation-fn relu \
       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
       --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
       --weight-decay 0.0001 --dropout 0.3 \
       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
       --max-tokens 40000 --patience 10 \
       --save-dir data/checkpoints/pretrain \
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
