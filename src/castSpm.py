
import  sentencepiece as spm
from    trainSpm import enModel, jaModel

###
# cd repMultipleReferences
# python3 ./src/castSpm.py
###

# pretrain
prefixPretrain = '/mntdir/tani/work/myFairseqScripts/multipleReferences/pretrain/data/sourceText/'

# finetune 
prefixFinetune = '/mntdir/tani/work/googletrans/data/revise/data/withSpToken/'


def applyBPE(model, srcPath, dstPath, textType):

    # 読み込み
    with open(srcPath, mode="r") as f:
        lines = f.readlines()

    out = ""
    for line in lines:

        line = line.strip()

        # 特殊トークンがある場合
        if textType=="src":
            spToken = line.split()[0]
            line = " ".join(line.split()[1:])
            out += f"{spToken} "

        # apply BPE
        tmp = model.EncodeAsPieces(line)
        out += " ".join(tmp) + "\n"
    
    # 書き込み
    with open(dstPath, mode="w") as f:
        f.write(out)
    

if __name__ == "__main__":
    
    # load
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load(f"{enModel}.model")

    sp_ja = spm.SentencePieceProcessor()
    sp_ja.Load(f"{jaModel}.model")


    print('applying...')

    srcPaths = [
        f"{prefixFinetune}train.src",
        f"{prefixFinetune}dev.src",
        f"{prefixFinetune}train.tgt",
        f"{prefixFinetune}dev.tgt",
        f"{prefixPretrain}MT/train.ja",
        f"{prefixPretrain}MT/train.en",
        f"{prefixPretrain}MT/dev.ja",
        f"{prefixPretrain}MT/dev.en",
        f"{prefixPretrain}MT/devtest.ja",
        f"{prefixPretrain}MT/devtest.en",
        "data/sourceText/combined/trainSpm.ja",
        "data/sourceText/combined/trainSpm.en",
        f"{prefixPretrain}test/test.src",
        f"{prefixPretrain}test/test.tgt"
    ]
    dstPaths = [
        "data/prepared/finetune/train.src",
        "data/prepared/finetune/dev.src",
        "data/prepared/finetune/train.tgt",
        "data/prepared/finetune/dev.tgt",
        "data/prepared/pretrain/train.ja",
        "data/prepared/pretrain/train.en",
        "data/prepared/pretrain/dev.ja",
        "data/prepared/pretrain/dev.en",
        "data/prepared/pretrain/devtest.ja",
        "data/prepared/pretrain/devtest.en",
        "data/prepared/combined/train.ja",
        "data/prepared/combined/train.en",
        "data/prepared/finetune/test.src",
        "data/prepared/finetune/test.tgt"
    ]


    for (srcPath, dstPath) \
        in zip(srcPaths, dstPaths):

        srcPathSuffix = srcPath.split(".")[-1]

        model = None
        if (srcPathSuffix == "src") or (srcPathSuffix == "ja"):
            model = sp_ja
        else :
            model = sp_en

        applyBPE(
            model,
            srcPath,
            dstPath,
            srcPathSuffix
            )
