
###
# # cd pretrain
# # python3 ./src/finetuneSetData.py
###
###
# cd repMultipleReferences
# python3 ./src/finetuneSetData.py
###

###
# .src
###
srcSrcPath = "data/prepared/finetune/train.src"
dstSrcPath = "data/prepared/finetune/setData/train.src"

###
# .tgt
###
srcTgtPath = "data/prepared/finetune/train.tgt"
dstTgtPath = "data/prepared/finetune/setData/train.tgt"


def sepSet(spTokens, tgtLines):
    """
    """
    setTgtText = ""
    lenSpTokens = len(spTokens)
    lenTgtLines = len(tgtLines)
    # print(f'lenSpTokens:{lenSpTokens}, lenTgtLines:{lenTgtLines}')

    tmp = []
    for (spToken, tgtLine) in zip(spTokens, tgtLines):
        tmp.append(f"{spToken.strip()} {tgtLine.strip()}")
    
    for i in range(lenSpTokens):
        setTgtText += " <sep> ".join(tmp) + "\n"

    return setTgtText


def makeSetData(srcLines, tgtLines):
    """
    """
    lenSrc = len(srcLines)
    lenTgt = len(tgtLines)
    print(f'lenSrc:{lenSrc}, lenTgt:{lenTgt}')

    exSpToken = ""
    exSrcSentence = ""
    exTmpSpTokens = []
    tmpSpTokens = []
    tmpTgtLines = []
    tgtOutput = ""

    for i, (srcLine, tgtLine) in enumerate(zip(srcLines, tgtLines)):

        # split srcLine 
        # into special token and srcSentence
        spToken     = srcLine.split()[0]
        srcSentence = " ".join(srcLine.split()[1:])

        exSpToken = spToken

        # set change
        if (exSrcSentence != srcSentence) \
            or ((spToken in tmpSpTokens) and (exSrcSentence==srcSentence)):

            # check
            if len(tmpSpTokens) > 5:
                print(f'{i} len(tmpSpTokens): {len(tmpSpTokens)}')
            
            # swap
            exSrcSentence = srcSentence
            exTmpSpTokens = tmpSpTokens

            # make set text
            setTgtText = sepSet(tmpSpTokens, tmpTgtLines)

            # add text
            tgtOutput += setTgtText.strip() + "\n"

            # init
            tmpSpTokens = []
            tmpTgtLines = []

        # append
        tmpSpTokens.append(spToken.strip())
        tmpTgtLines.append(tgtLine.strip())

        # set end
        if (i == lenSrc-1):
            # make set text
            setTgtText = sepSet(tmpSpTokens, tmpTgtLines)

            # add text
            tgtOutput += setTgtText.strip() + "\n"
    
    return tgtOutput.strip()

###
# execute
###
if __name__ == "__main__":

    # read
    with open(srcSrcPath, mode="r") as f:
        srcText     = f.read()

    with open(srcSrcPath, mode="r") as f:
        srcLines    = f.readlines()

    with open(srcTgtPath, mode="r") as f:
        tgtLines = f.readlines()

    ###
    # <sep> set data
    ###
    tgtOutput = makeSetData(srcLines, tgtLines)

    print(f'len(tgtOutput) : {len(tgtOutput.splitlines())}')

    ###
    # write
    ###
    with open(dstSrcPath, mode="w") as f:
        f.write(srcText)

    with open(dstTgtPath, mode="w") as f:
        f.write(tgtOutput)
