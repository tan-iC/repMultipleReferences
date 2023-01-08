from    preprocess\
    import preprocessSentences, preprocessTaggedSentences, preprocessSetSentences


###
# cd repMultipleReferences
# python3 ./src/total32k/preprocessFinetune.py
###


# model name to decode/encode
enModelPath = "data/spm/en32k.model"
jaModelPath = "data/spm/ja32k.model"
total32kModelPath = 'total32k/data/spm/total32k.model'

# pretrain src/dst path
baseSrcPath = 'data/prepared/finetune'
baseDstPath = 'total32k/data/prepared/finetune'

# set src/tgt text
setSrcPaths = [
    'data/prepared/finetune/setData/train.src',
    'data/prepared/finetune/setData/train.tgt'
]

setDstPaths = [
    'total32k/data/prepared/finetune/setData/train.src',
    'total32k/data/prepared/finetune/setData/train.tgt'
]

# types
textTypes = [
    "dev",
    "test"
]

# languages
langs = [
    "src",
    "tgt"
]

# model paths
decodeModelPaths = [
    jaModelPath,
    enModelPath
]


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

            if lang == langs[0]:
                # src : tagged sentence
                encoded = preprocessTaggedSentences(
                    currentSrcPath,
                    decodeModelPath,
                    total32kModelPath
                    )
            else:
                # tgt : sentence
                encoded = preprocessSentences(
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

    ###
    # preprocess set data
    ###
    print(f"{setSrcPaths[0]}, {setSrcPaths[1]} preprocessing ...")
    encodedSrc, encodedTgt =\
        preprocessSetSentences(
            setSrcPaths[0], 
            setSrcPaths[1], 
            decodeModelPaths[0], 
            decodeModelPaths[1],
            total32kModelPath
            )

    ###
    # write
    ###
    print("writing ...")
    with open(setDstPaths[0], mode="w") as f:
        f.write(encodedSrc)

    print("writing ...")
    with open(setDstPaths[1], mode="w") as f:
        f.write(encodedTgt)
