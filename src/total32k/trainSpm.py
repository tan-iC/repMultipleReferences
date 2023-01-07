
import  sentencepiece as spm
from    tqdm import tqdm

###
# cd repMultipleReferences
# python3 ./src/total32k/trainSpm.py
###

# vocab size
vocab_size = 32000

# model name
enModelPath = "data/spm/en32k"
jaModelPath = "data/spm/ja32k"
total32kModel = 'total32k/data/spm/total32k'

# combined data : en
enSrcCombinedPath   = 'data/sourceText/combined/trainSpm.en'

# combined data : ja
jaSrcCombinedPath   = 'data/sourceText/combined/trainSpm.ja'

# combined data : en + ja
dstCombinedPath = 'total32k/data/combined.txt'


def decodeSpmPairs(text : str, model):
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


def decodedData(srcPath, modelPath):
    """
    read text, load spm model and decode

    input
        - srcPath : str, src text path
        - modelPath: str, spm model path
    """
    with open(srcPath, mode="r") as f:
        srcText = f.read()
    
    # load sentencepiece model
    model = spm.SentencePieceProcessor()
    model.Load(modelPath)

    print(f'{modelPath} vocab size: {model.GetPieceSize()}')

    # decode
    decoded = decodeSpmPairs(srcText, model)

    return decoded


def trainSpm(combinedData : str, modelName : str, characterCovarage=0.9995, vocabSize=32000):
    """
    """
    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(f'--input={combinedData} --model_prefix={modelName} --character_coverage={characterCovarage} --vocab_size={vocabSize} --train_extremely_large_corpus=true')


if __name__ == "__main__":

    print("reading ...")

    ###
    # en
    ###
    # decodedEn = decodedData(enSrcCombinedPath, enModelPath)
    with open(enSrcCombinedPath, mode="r") as f:
        enText = f.read()

    print("reading ...")

    ###
    # ja
    ###
    # decodedJa = decodedData(jaSrcCombinedPath, jaModelPath)
    with open(jaSrcCombinedPath, mode="r") as f:
        jaText = f.read()

    ###
    # combined
    ###
    # combinedText = decodedEn.strip() + '\n' + decodedJa.strip()
    combinedText = enText.strip() + "\n" + jaText.strip()

    print("writing ...")

    # write
    with open(dstCombinedPath, mode="w") as f:
        f.write(combinedText)
    
    print("training ...")

    # train spm
    trainSpm(dstCombinedPath, total32kModel)
