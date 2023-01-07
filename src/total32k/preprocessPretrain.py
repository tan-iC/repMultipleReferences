
import  sentencepiece as spm
from    tqdm import tqdm

###
# cd repMultipleReferences
# python3 ./src/total32k/preprocessPretrain.py
###

# model name to decode/encode
enModelPath = "data/spm/en32k.model"
jaModelPath = "data/spm/ja32k.model"
total32kModelPath = 'total32k/data/spm/total32k.model'

# pretrain src/dst path
baseSrcPath = 'data/prepared/pretrain'
baseDstPath = 'total32k/data/prepared/pretrain'

# combined en/ja text
combSrcPaths = [
    'data/prepared/combined/train.en',
    'data/prepared/combined/train.ja'
]

combDstPaths = [
    'total32k/data/prepared/combined/train.en',
    'total32k/data/prepared/combined/train.ja'
]

# types
textTypes = [
    "dev",
    "devtest",
    "train"
]

# languages
langs = [
    "en",
    "ja"
]

# model paths
decodeModelPaths = [
    enModelPath,
    jaModelPath
]


# 
def decodeSentences(text : str, model):
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
def encodeSentences(text : str, model):
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
def preprocessPretrain(srcPath, decodeModelPath, encodeModelPath):
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

    decoded = decodeSentences(srcText, decodeModel)

    ###
    # encode
    ###
    encodeModel = spm.SentencePieceProcessor()
    encodeModel.Load(encodeModelPath)
    print(f'{encodeModelPath} vocab size: {encodeModel.GetPieceSize()}')

    encoded = encodeSentences(decoded, encodeModel)

    return encoded


if __name__ == "__main__":

    for textType in textTypes:

        for (lang, decodeModelPath) in zip(langs, decodeModelPaths):

            ###
            # preprocess
            ###
            print(f"{textType}.{lang} preprocessing ...")
            currentSrcPath = f"{baseSrcPath}/{textType}.{lang}"
            # encoded = preprocessPretrain(
            #     currentSrcPath,
            #     decodeModelPath,
            #     total32kModelPath
            #     )


            # ###
            # # write
            # ###
            # print("writing ...")
            # currentDstPath = f"{baseDstPath}/{textType}.{lang}"
            # with open(currentDstPath, mode="w") as f:
            #     f.write(encoded)

    ###
    # combined
    ###
    for (combSrcPath, combDstPath) in zip(combSrcPaths, combDstPaths):
        print(f"{combSrcPath} preprocessing ...")
        encoded = preprocessPretrain(
            combSrcPath,
            decodeModelPath,
            total32kModelPath
            )

        print("writing ...")
        with open(combDstPath, mode="w") as f:
            f.write(encoded)
