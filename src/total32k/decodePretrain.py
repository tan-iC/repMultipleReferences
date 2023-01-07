
import  sentencepiece as spm
from    tqdm import tqdm

###
# cd repMultipleReferences
# python3 ./src/total32k/decodePretrain.py
###

# model name to decode/encode
enModelPath = "data/spm/en32k"
jaModelPath = "data/spm/ja32k"
total32kModel = 'total32k/data/spm/total32k'

# pretrain src/dst path
baseSrcPath = 'data/prepared/pretrain'
baseDstPath = 'total32k/data/prepared/pretrain'

# combined en/ja text
combEnPath = 'data/combined/train.en'
combJaPath = 'data/combined/train.ja'

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
def decodedData(srcPath, decodeModelPath, encodeModelPath):
    """
    read text, load spm model and decode

    input
        - srcPath : str, src text path
        - modelPath: str, spm model path
    """
    with open(srcPath, mode="r") as f:
        srcText = f.read()
    
    # load sentencepiece decode model
    decodeModel = spm.SentencePieceProcessor()
    decodeModel.Load(decodeModelPath)

    print(f'{decodeModelPath} vocab size: {decodeModel.GetPieceSize()}')

    # decode
    decoded = decodeSentences(srcText, decodeModel)

    # load sentencepiece encode model
    encodeModel = spm.SentencePieceProcessor()
    encodeModel.Load(encodeModelPath)

    print(f'{encodeModelPath} vocab size: {encodeModel.GetPieceSize()}')


    return decoded


if __name__ == "__main__":

    print("reading ...")


    print("writing ...")
