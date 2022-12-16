
import  sentencepiece as spm

###
# cd repMultipleReferences
# python3 ./src/trainSpm.py
###

# vocab size
vocab_size = 32000

# model name
enModel = "data/spm/en32k"
jaModel = "data/spm/ja32k"

# pretrain data : en
enPretrainPath   = '/mntdir/tani/work/myFairseqScripts/multipleReferences/pretrain/data/sourceText/MT/train.en'

# pretrain data : ja
jaPretrainPath   = '/mntdir/tani/work/myFairseqScripts/multipleReferences/pretrain/data/sourceText/MT/train.ja'

# finetune data : en
enFinetunePath   = '/mntdir/tani/work/googletrans/data/revise/data/raw/train.en'

# finetune data : ja
jaFinetunePath   = '/mntdir/tani/work/googletrans/data/revise/data/raw/train.ja'

# combined data : en
enCombinedPath    = 'data/sourceText/combined/trainSpm.en'

# combined data : ja
jaCombinedPath    = 'data/sourceText/combined/trainSpm.ja'


def trainSpm():

    with open(enFinetunePath, mode="r") as f:
        enFinetune = f.read()

    with open(enPretrainPath, mode="r") as f:
        enPretrain = f.read()

    with open(jaFinetunePath, mode="r") as f:
        jaFinetune = f.read()

    with open(jaPretrainPath, mode="r") as f:
        jaPretrain = f.read()


    # combined data : en
    enCombined  = enFinetune.strip() + '\n' + enPretrain.strip()
    print(f'len(enFinetune.splitlines()): {len(enFinetune.splitlines())}')
    print(f'len(enPretrain.splitlines()): {len(enPretrain.splitlines())}')
    print(f'len(enCombined.splitlines()): {len(enCombined.splitlines())}')

    with open(enCombinedPath, mode="w") as f:
        f.write(enCombined)


    # ja combined data
    jaCombined  = jaFinetune.strip() + '\n' + jaPretrain.strip()
    print(f'len(jaFinetune.splitlines()): {len(jaFinetune.splitlines())}')
    print(f'len(jaPretrain.splitlines()): {len(jaPretrain.splitlines())}')
    print(f'len(jaCombined.splitlines()): {len(jaCombined.splitlines())}')

    with open(jaCombinedPath, mode="w") as f:
        f.write(jaCombined)


    # learning
    print('learning...')
    spm.SentencePieceTrainer.Train(
        f'--input={enCombinedPath} --model_prefix={enModel} --vocab_size={vocab_size} --character_coverage=1'
    )
    spm.SentencePieceTrainer.Train(
        f'--input={jaCombinedPath} --model_prefix={jaModel} --vocab_size={vocab_size} --character_coverage=0.9998'
    )


if __name__ == "__main__":
    trainSpm()
