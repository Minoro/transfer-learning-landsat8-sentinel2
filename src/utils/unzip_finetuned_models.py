import zipfile

MODELS_ZIPED = '../../resources/best_models_manual_annotations.zip'

OUTPUT_PATH = '../../resources/transfer_learning/weights'

if __name__ == '__main__':

    print('Unziping fine-tuned models...')

    with zipfile.ZipFile(MODELS_ZIPED) as zip:
        zip.extractall(OUTPUT_PATH)

    print('Done!')

