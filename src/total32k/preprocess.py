
import  sentencepiece as spm
from    tqdm import tqdm


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
def preprocessSentences(srcPath, decodeModelPath, encodeModelPath):
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


###
# tagged sentences preprocess
###
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
        pieces   = line.strip().split()

        # sptoken
        spToken = pieces[0]

        # decode pieces to sentence
        decoded = model.DecodePieces(pieces[1:])
        decodedText += f"{spToken} {decoded}\n"

    return decodedText.strip()


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
        # sptoken
        sptoken, sentence = sepalateSpToken(line)

        # encode sentence as pieces
        pieces = model.EncodeAsPieces(sentence)
        encoded = " ".join(pieces)
        encodedText += f'{sptoken} {encoded}\n'

    return encodedText.strip()


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


###
# set sentences preprocess
###
def sepalateSpToken(line : str):
    """
    """
    # split by space
    tokens = line.strip().split()

    # sepalate special token and sentence
    spToken = tokens[0]
    sentence = " ".join(tokens[1:]).strip()

    return spToken, sentence


def decodeSetSentences(
    srcText : str, 
    tgtText : str, 
    srcModel, 
    tgtModel,
    sepToken
    ):
    """
    """

    srcLines = srcText.splitlines()
    tgtLines = tgtText.splitlines()

    decodedSrc = ""
    decodedTgt = ""
    exSrc = ""

    for (srcLine, tgtLine) in tqdm(zip(srcLines, tgtLines)):

        # split by space
        srcTokens   = srcLine.strip().split()

        # decode: src sentence
        srcSpToken  = srcTokens[0]
        srcSentence = srcModel.DecodePieces(srcTokens[1:])

        # different set
        if exSrc != srcSentence:

            # split by <sep> token
            tgtTokensSet = tgtLine.strip().split(sepToken)

            for tgtTokens in tgtTokensSet:

                # split by space
                tgtTokens   = tgtTokens.strip().split()

                # decode tgt sentence
                tgtSpToken  = tgtTokens[0]
                tgtSentence = tgtModel.DecodePieces(tgtTokens[1:])

                # output
                decodedSrc += f"{tgtSpToken} {srcSentence}\n"
                decodedTgt += f"{tgtSentence}\n"
        
        # swap
        exSrc = srcSentence

    return decodedSrc.strip(), decodedTgt.strip()


def encodeSetSentences(
    srcText : str, 
    tgtText : str, 
    model,
    sepToken
    ):
    """
    """
    srcLines = srcText.splitlines()
    tgtLines = tgtText.splitlines()

    encodedSrc = ""
    encodedTgt = ""
    exSrc = ""

    for (srcLine, tgtLine) in tqdm(zip(srcLines, tgtLines)):
        ###
        # encode: src sentence
        ###

        # sepalate special token and sentence
        srcSpToken, srcSentence = sepalateSpToken(srcLine)

        # encode src sentence as pieces
        srcPieces = model.EncodeAsPieces(srcSentence)

        # different set
        if exSrc != srcSentence:

            # split by <sep> token
            tgtTokensSet = tgtLine.strip().split(sepToken)

            for tgtTokens in tgtTokensSet:
                ###
                # decode: tgt sentence
                ###

                # sepalate special token and sentence
                tgtSpToken, tgtSentence = sepalateSpToken(tgtTokens)

                # encode tgt sentence as pieces
                tgtPieces = model.EncodeAsPieces(tgtSentence)

                ###
                # output
                ###
                encodedSrc += f"{srcSpToken} {' '.join(srcPieces).strip()}\n"
                encodedTgt += f"{' '.join(tgtPieces).strip()}\n"
        
        # swap
        exSrc = srcSentence

    return encodedSrc.strip(), encodedTgt.strip()


def turnSetSentences(
    srcText : str, 
    tgtText : str, 
    srcModel, 
    tgtModel,
    totalModel,
    sepToken : str
    ):
    """
    """

    srcLines = srcText.splitlines()
    tgtLines = tgtText.splitlines()

    turnedSrc = ""
    turnedTgt = ""
    exSrc = ""

    for (srcLine, tgtLine) in tqdm(zip(srcLines, tgtLines)):

        # split by space
        srcTokens   = srcLine.strip().split()

        # decode src sentence
        srcSpToken  = srcTokens[0]
        decodedSrc = srcModel.DecodePieces(srcTokens[1:]).strip()

        # encode src sentence as pieces
        srcPieces = totalModel.EncodeAsPieces(decodedSrc)
        encodedSrc = " ".join(srcPieces).strip()

        # different set
        if exSrc != decodedSrc:

            # split by <sep> token
            tgtTokensSet = tgtLine.strip().split(sepToken)

            tgtSentences = []

            for tgtTokens in tgtTokensSet:

                # split by space
                tgtTokens   = tgtTokens.strip().split()

                # decode tgt sentence
                tgtSpToken  = tgtTokens[0]
                decodedTgt = tgtModel.DecodePieces(tgtTokens[1:]).strip()

                # encode tgt sentence as pieces
                tgtPieces = totalModel.EncodeAsPieces(decodedTgt)
                encodedTgt = " ".join(tgtPieces).strip()

                # keep
                tgtSentences.append(f'{tgtSpToken} {encodedTgt}')

                # output
                turnedSrc += f"{tgtSpToken} {encodedSrc}\n"
            
            tmp = f' {sepToken} '.join(tgtSentences).strip()
            for i in range(len(tgtSentences)):
                # output
                turnedTgt += f"{tmp}\n"
        
        # swap
        exSrc = decodedSrc

    return turnedSrc.strip(), turnedTgt.strip()


def preprocessSetSentences(
    srcTextPath : str, 
    tgtTextPath : str, 
    srcModelPath : str, 
    tgtModelPath : str, 
    totalModelPath : str,
    sepToken="<sep>"
    ):
    """
    """

    # src text and spm model
    with open(srcTextPath, mode="r") as f:
        srcText = f.read()
    srcModel = spm.SentencePieceProcessor()
    srcModel.Load(srcModelPath)

    # tgt text and spm model
    with open(tgtTextPath, mode="r") as f:
        tgtText = f.read()
    tgtModel = spm.SentencePieceProcessor()
    tgtModel.Load(tgtModelPath)

    # total spm model
    totalModel = spm.SentencePieceProcessor()
    totalModel.Load(totalModelPath)


    ###
    # decode and encode
    ###
    encodedSrc, encodedTgt =\
        turnSetSentences(srcText, tgtText, srcModel, tgtModel, totalModel, sepToken)

    return encodedSrc, encodedTgt
