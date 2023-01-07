
import  sentencepiece as spm
from    tqdm import tqdm
# from    


###
# cd repMultipleReferences
# python3 ./src/total32k/preprocessFinetune.py
###


# model name to decode/encode
enModelPath = "data/spm/en32k.model"
jaModelPath = "data/spm/ja32k.model"
total32kModelPath = 'total32k/data/spm/total32k.model'

# pretrain src/dst path
baseSrcPath = 'data/prepared/pretrain'
baseDstPath = 'total32k/data/prepared/pretrain'

# combined en/ja text
setSrcPaths = [
    'data/combined/train.en',
    'data/combined/train.ja'
]

setDstPaths = [
    'total32k/data/prepared/combined/train.en',
    'total32k/data/prepared/combined/train.ja'
]

# types
textTypes = [
    "dev",
    "test",
    "train"
]

# languages
langs = [
    "src",
    "tgt"
]

# model paths
decodeModelPaths = [
    enModelPath,
    jaModelPath
]


# 
def decodeTaggedSentences(text : str, model):
    """
    input
        - text : str, encoded text
        - model : , spm model

    output
        - decoded text
    """

    decodedText = ""
    lines = text.strip().splitlines()
    for line in tqdm(lines):            
        # split: by space
        tokens   = line.strip().split()

        # decode: sentence
        sentence = model.DecodePieces(tokens)

        decodedText += sentence.strip() + '\n'

    return decodedText.strip()

# 
def encodeTaggedSentences(text : str, model):
    """
    input
        - text : str, encoded text
        - model : , spm model

    output
        - encoded text
    """

    encodedText = ""
    lines = text.strip().splitlines()
    for line in tqdm(lines):            

        # encode: line
        tokens = model.EncodeAsPieces(line.strip())
        sentence = " ".join(tokens)
        encodedText += sentence.strip() + '\n'

    return encodedText.strip()


# 
def preprocessTaggedSentences(srcPath, decodeModelPath, encodeModelPath):
    """
    read text, load spm model and decode

    input
        - srcPath : str, src text path
        - modelPath: str, spm model path
    """
    with open(srcPath, mode="r") as f:
        srcText = f.read()

    ###
    # decode
    ###
    decodeModel = spm.SentencePieceProcessor()
    decodeModel.Load(decodeModelPath)
    print(f'{decodeModelPath} vocab size: {decodeModel.GetPieceSize()}')

    decoded = decodeTaggedSentences(srcText, decodeModel)

    ###
    # encode
    ###
    encodeModel = spm.SentencePieceProcessor()
    encodeModel.Load(encodeModelPath)
    print(f'{encodeModelPath} vocab size: {encodeModel.GetPieceSize()}')

    encoded = encodeTaggedSentences(decoded, encodeModel)

    return encoded


#
def decodeSetSentences(srcLines, tgtLines, srcModel, tgtModel, sepToken="<sep>"):
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


###
# execute
###
if __name__ == "__main__":

    for textType in textTypes:

        for (lang, decodeModelPath) in zip(langs, decodeModelPaths):

            ###
            # preprocess
            ###
            print(f"{textType}.{lang} preprocessing ...")
            currentSrcPath = f"{baseSrcPath}/{textType}.{lang}"
            encoded = preprocessFinetune(
                currentSrcPath,
                decodeModelPath,
                total32kModelPath
                )


            ###
            # write
            ###
            print("writing ...")
            currentDstPath = f"{baseDstPath}/{textType}.{lang}"
            with open(currentDstPath, mode="w") as f:
                f.write(encoded)
