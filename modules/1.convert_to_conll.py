"""
Iterate thru all txt files and find matching *.out files
Load, convert to conll and save as conll to specified dir.
"""
import os
from pathlib import Path
from Document import Document, Entity




def read_ann_save_conll(txt_file, ann_file, conll_file):
    doc = Document(txt_file)
    doc.load_entities_from_ann(ann_file)
    doc.save_as_conll(conll_file)



def run_converter(txtdir, annotationsdir, conlldir):
    for subdir, dirs, files in os.walk(txtdir):
        for filename in files:
            txt_file = os.path.join(subdir, filename)
            ann_file = os.path.join(subdir.replace(txtdir, annotationsdir), ".out".join(filename.rsplit('.txt', 1)))#filename.replace(".txt",".ann")
            conll_file = os.path.join(subdir.replace(txtdir, conlldir), ".conll".join(filename.rsplit('.txt', 1)))
            conll_out_dir = subdir.replace(txtdir, conlldir)
            os.makedirs(conll_out_dir, exist_ok=True)
            read_ann_save_conll(txt_file, ann_file, conll_file)


if __name__ == "__main__":
    # read_ann_save_conll(
    #     "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\conlldir\\asia_bibi_bg.txt_file_1.txt" ,
    #     "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\conlldir\\asia_bibi_bg.txt_file_1.out",
    #     "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\conlldir\\asia_bibi_bg.txt_file_1.conll"
    # )
    # read_ann_save_conll(
    #     "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\raw\\ryanair\\ru\\ryan-ru_file_124.txt",
    #     "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\annotated\\ryanair\\ru\\ryan-ru_file_124.out",
    #     "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\conlldir\\ryan-ru_file_124.conll"
    # )
    txtdir = "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\raw"
    annotationsdir = "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\annotated"
    conlldir = "D:\\NER\\BSNLP_2021\\data\\BSNLP\\bsnlp2021_train_r1\\conlldir"
    run_converter(txtdir, annotationsdir, conlldir)
