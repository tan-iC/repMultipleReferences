
import  sentencepiece as spm
from    tqdm import tqdm

###
# cd googletrans/src
# python3 MakeDetokenizeData.py
###

# encoded text : train
jaTrainPath = "../data/filtered/4finetune/my_method_01/1006/train.ja"
enTrainPath = "../data/filtered/4finetune/my_method_01/1006/train.en"

# encoded text : valid
jaValidPath = "../data/filtered/4finetune/my_method_01/1006/dev.ja"
enValidPath = "../data/filtered/4finetune/my_method_01/1006/dev.en"

# copy
jaTrainCopy = "../data/revise/data/sourceText/train.ja"
enTrainCopy = "../data/revise/data/sourceText/train.en"
jaValidCopy = "../data/revise/data/sourceText/dev.ja"
enValidCopy = "../data/revise/data/sourceText/dev.en"

# sentencepiece model path : ja
jaModelPath = "sentencepiece_ja_.model"

# sentencepiece model path : en
enModelPath = "sentencepiece_en_.model"

# decoded text with special token : train
jaTrainWspToken = "../data/revise/data/withSpToken/train.src"
enTrainWspToken = "../data/revise/data/withSpToken/train.tgt"

# decoded text with special token : valid
jaValidWspToken = "../data/revise/data/withSpToken/dev.src"
enValidWspToken = "../data/revise/data/withSpToken/dev.tgt"

# decoded text without special token : train
jaTrainWOspToken = "../data/revise/data/raw/train.src"
enTrainWOspToken = "../data/revise/data/raw/train.tgt"

# decoded text without special token : valid
jaValidWOspToken = "../data/revise/data/raw/dev.src"
enValidWOspToken = "../data/revise/data/raw/dev.tgt"


def decodeSpm(srcLines, tgtLines, srcModel, tgtModel, sepToken="<sep>"):
    """
    srcLines: List[str]
    tgtLines: List[str]
    srcModel: spm
    tgtModel: spm
    """
    
    srcOutputTextWspToken = ""
    srcOutputText   = ""
    tgtOutputText   = ""
    exSrcSentence   = ""

    for (srcLine, tgtLine) in tqdm(zip(srcLines, tgtLines)):

        # split by space
        srcTokens   = srcLine.strip().split()

        # decode: src sentence
        srcSpToken  = srcTokens[0]
        srcSentence = srcModel.DecodePieces(srcTokens[1:])

        # different set
        if exSrcSentence != srcSentence:

            # split by <sep> token
            tgtTokensSet = tgtLine.strip().split(sepToken)

            for tgtTokens in tgtTokensSet:

                # split by space
                tgtTokens   = tgtTokens.strip().split()

                # decode tgt sentence
                tgtSpToken  = tgtTokens[0]
                tgtSentence = tgtModel.DecodePieces(tgtTokens[1:])

                # output
                srcOutputTextWspToken += f"{tgtSpToken} {srcSentence}\n"
                srcOutputText += f"{srcSentence}\n"
                tgtOutputText += f"{tgtSentence}\n"
        
        # swap
        exSrcSentence = srcSentence

    return srcOutputTextWspToken, srcOutputText, tgtOutputText


if __name__ == "__main__":


    # read
    with open(jaTrainPath, mode="r") as f:
        jaTrain = f.read()
    
    jaTrainLines = jaTrain.splitlines()
    print(f'len(jaTrainPath): {len(jaTrainLines)}')

    with open(enTrainPath, mode="r") as f:
        enTrain = f.read()

    enTrainLines = enTrain.splitlines()
    print(f'len(enTrainPath): {len(enTrainLines)}')

    with open(jaValidPath, mode="r") as f:
        jaValid = f.read()

    jaValidLines = jaValid.splitlines()
    print(f'len(jaValidPath): {len(jaValidLines)}')

    with open(enValidPath, mode="r") as f:
        enValid = f.read()

    enValidLines = enValid.splitlines()
    print(f'len(enValidPath): {len(enValidLines)}')


    # copy
    with open(jaTrainCopy, mode="w") as f:
        f.write(jaTrain)

    with open(enTrainCopy, mode="w") as f:
        f.write(enTrain)

    with open(jaValidCopy, mode="w") as f:
        f.write(jaValid)

    with open(enValidCopy, mode="w") as f:
        f.write(enValid)


    # load sentencepiece model : ja
    jaModel = spm.SentencePieceProcessor()
    jaModel.Load(jaModelPath)
    print(f'ja model vocab size: {jaModel.GetPieceSize()}')

    # load sentencepiece model : en
    enModel = spm.SentencePieceProcessor()
    enModel.Load(enModelPath)
    print(f'en model vocab size: {enModel.GetPieceSize()}')


    # decode train text
    jaTrainOutputWspToken, jaTrainOutput, enTrainOutput \
        = decodeSpm(jaTrainLines, enTrainLines, jaModel, enModel)

    # decode valid text
    jaValidOutputWspToken, jaValidOutput, enValidOutput \
        = decodeSpm(jaValidLines, enValidLines, jaModel, enModel)


    # write data with special token
    with open(jaTrainWspToken, mode="w") as f:
        f.write(jaTrainOutputWspToken)

    with open(jaValidWspToken, mode="w") as f:
        f.write(jaValidOutputWspToken)

    with open(enTrainWspToken, mode="w") as f:
        f.write(enTrainOutput)

    with open(enValidWspToken, mode="w") as f:
        f.write(enValidOutput)


    # write data without special token
    with open(jaTrainWOspToken, mode="w") as f:
        f.write(jaTrainOutput)

    with open(jaValidWOspToken, mode="w") as f:
        f.write(jaValidOutput)

    with open(enTrainWOspToken, mode="w") as f:
        f.write(enTrainOutput)

    with open(enValidWOspToken, mode="w") as f:
        f.write(enValidOutput)
