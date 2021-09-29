import os
from shutil import copyfile
from pathlib import Path


def get_sentences_from_lines(lines):
    sentences = []
    string = ''
    for line in lines:
        if line.strip():
            string = string + line
            continue
        # Empty line is sentence separator in conll-2003 files
        sentences.append(string)
        string = ''
    return sentences


def get_splits(source_sentences, fold, NUM_FOLDS):
    test_sents = []
    dev_sents = []
    train_sents = []
    # fold = 1: test=1,dev=2,train=rest
    # fold = NUM_FOLDS: test=NUM_FOLDS, dev=1, train=rest
    dev_fold = 0 if fold == NUM_FOLDS-1 else fold + 1
    for sentence_num, source_sentence in enumerate(source_sentences):
        remainder = sentence_num % NUM_FOLDS
        if remainder == fold:
            test_sents.append(source_sentence)
        elif remainder == dev_fold:
            dev_sents.append(source_sentence)
        else:
            train_sents.append(source_sentence)
    return (test_sents, dev_sents, train_sents)


def main(src_data_dir, out_data_dir, NUM_FOLDS):
    source_file = src_data_dir.joinpath("joined.txt")
    labels_file = src_data_dir.joinpath("labels.txt")
    with open(source_file, 'r', encoding='utf-8') as infile:
        source_lines = infile.readlines()   # Assume not-too-big file
    source_sentences = get_sentences_from_lines(source_lines)

    for fold in range(NUM_FOLDS):
        fold_output_dir = out_data_dir.joinpath(str(fold))
        os.makedirs(fold_output_dir, exist_ok =True)
        copyfile(labels_file, fold_output_dir.joinpath("labels.txt"))

        test_sents, dev_sents, train_sents = get_splits(source_sentences, fold, NUM_FOLDS)
        with open(fold_output_dir.joinpath("test.txt"), 'w', encoding='utf-8') as outfile:
            for test_sent in test_sents:
                outfile.write("{}\n".format(test_sent))
        with open(fold_output_dir.joinpath("dev.txt"), 'w', encoding='utf-8') as outfile:
            for dev_sent in dev_sents:
                outfile.write("{}\n".format(dev_sent))
        with open(fold_output_dir.joinpath("train.txt"), 'w', encoding='utf-8') as outfile:
            for train_sent in train_sents:
                outfile.write("{}\n".format(train_sent))


if __name__ == "__main__":
    # src_data_dir must contain labels.txt and train.txt - this file will be split
    # number of folds. one will become test.txt, one dev.txt and the rest train.txt
    # default 10, minimum 3
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="input data dir, to split into folds")
    parser.add_argument("--num_folds", default=10, help="number of folds")
    args = parser.parse_args()
    
    # src_data_dir = Path(r"E:\NER\Data\Daiga_BOT2\data_with_test")
    src_data_dir = Path(args.src_dir)
    out_data_dir = src_data_dir

    print("Starting:")
    print(f"NUM_FOLDS: {args.num_folds}")
    print(f"source_dir: {src_data_dir}")
    print(f"output_dir: {out_data_dir}")

    main(src_data_dir, out_data_dir, args.num_folds)