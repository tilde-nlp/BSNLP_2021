import os
from pathlib import Path 
from Conll_2003 import Conll2003
import argparse

def concat(proj_dir, concat_file):
    with open(concat_file, 'w', encoding='utf-8') as concatf:
        for subdir, dirs, filenames in os.walk(proj_dir):
            for filename in filenames:
                if not filename.endswith('.conll'):
                    continue
                filepath = Path(subdir).joinpath(filename)
                with open(filepath, 'r', encoding='utf-8') as conll:
                    for line in conll:
                        concatf.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concat conll files.')
    parser.add_argument('source_dir', help='Directory with files')
    parser.add_argument('target_file', help='Filename for concatenated conll')
    args = parser.parse_args()
    source_dir = args.source_dir
    target_file = args.target_file

    concat(source_dir, target_file)

    # Create "labels" file
    conlldoc = Conll2003()
    conlldoc.load_from_file(target_file)
    conlldoc.update_labels_file(target_file)

