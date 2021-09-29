"""
    http://bsnlp.cs.helsinki.fi/shared-task.html
    The 3rd edition of the shared task covers six languages:

        Bulgarian,
        Czech,
        Polish,
        Russian,
        Slovene,
        Ukrainian.

    and five types of named entities:

        persons,        PER people,families, fictive persons -titles, honorifics,functions “​CEO Dr. Jan Kowalski​”->​Jan Kowalski​
                                Toponym-based groups of ppl (Latvians, Europeans), groups of ppl from organizations
                                but no muslims (have no organization above them)
        locations,      LOC toponyms, facilities, country names (referring to GPE), 
        organizations,  ORG all kinds of organizations, seats of organizations with locations
        events,         EVT named mentions of evts, future,speculative evts, "winter olympics in canda" = evt
        products.       PRO product names, names of legal documents

    Evaluation is case-insensitive
    Lemma should be given for surface form: UE->UE and UniiEuropejskiej->UniaEuropejska
    For complex NEs only longest should be recognized

    INPUT:
        documents containing metadata + txt: 
        First 5 lines <ID><LANGUAGE><CREATION-DATE><URL><TITLE>
        Sixt line is document text
        note that both​ <CREATION-DATE> and ​<TITLE> information might be missing, then lines are empty
    OUTPUT:
        Corresponding file to each input file with format:
        The first line should contain only the ID of the file in the test corpus
        Each subsequent line should be of the format:Named-entity-mention<TAB>base-form<TAB>category<TAB>cross-lingual ID
        The files with system response should be ​encoded using UTF-8​ encoding.
        A file containing a system response
        should have the same name as the corresponding input file with an ​additional extension “.out”​.


    Micro/Macro Precision, F1
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    Example precision Pr=TP/(TP+FP)
    Let's imagine you have a One-vs-All (there is only one correct class output per example)
    multi-class classification system with four classes and the following numbers when tested:
        Class A: 1 TP and 1 FP
        Class B: 10 TP and 90 FP
        Class C: 1 TP and 1 FP
        Class D: 1 TP and 1 FP

    You can see easily that PrA=PrC=PrD=0.5, whereas PrB=0.1

        A macro-average will then compute: Pr=(0.5+0.1+0.5+0.5)/4=0.4
        A micro-average will compute: Pr=(1+10+1+1)/(2+100+2+2)=0.123

"""
import sys
import argparse
import os
sys.path.insert(0, './modules')
from modules.BERT_NER_PRED import *
from modules.Document import Entity, Document
# Entity = namedtuple('Entity', 'text lemma type linked_name')
from sentence_transformers import SentenceTransformer , util
import math
from collections import namedtuple
from pathlib import Path
import torch

# main("bg", "", "", "Asia Bibi lives in Latvia.", "saved_model/")

# 1. Iterate over test files, and load each to ''Document''
# 2. Do NER on each document + lemmatizer
# 3. For all entities in document do linking
# 4. Save ann files

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

KBEntity = namedtuple('KBEntity', 'text e_type linked_name embedding')

class Linker:
    def __init__(self, kb_file = "KB.txt"):
        self.kb_file = kb_file
        # self.kb = self.load_kb_from_file(self.kb_file) # Tensor save/load does not work maybe will use pickle.
        self.kb = []
        self.embedder = SentenceTransformer('LaBSE')


    # def load_kb_from_file(self, filename):
    #     kb = []
    #     if not os.path.exists(filename):
    #         return kb
    #     with open(filename, 'r', encoding='utf-8') as kbfile:
    #         for line in kbfile:
    #             line = line.strip()
    #             txt, e_type, e_id, emb = line.split("\t")
    #             kbentity = KBEntity(txt, e_type, e_id, emb)
    #             kb.append(kbentity)
    #     return kb

    # def save_kb_to_file(self, filename):
    #     with open(filename, 'w', encoding='utf-8') as kbfile:
    #         for kbent in self.kb:
    #             kbfile.write("{}\t{}\t{}\t{}\n".format(kbent.text, kbent.e_type, kbent.linked_name, kbent.embedding))
    
    def get_id(self, text, e_type):
        embedding = self.embedder.encode([text], convert_to_tensor=True, show_progress_bar=False)[0]
        max_score = 0
        best_match = None
        embeddings = []
        for kbent in self.kb:
            embeddings.append(torch.FloatTensor(kbent.embedding))
        if not len(embeddings) == 0:
            t_a = torch.FloatTensor(embedding)
            t_b = torch.stack((embeddings),dim = 0)
            similarities = util.pytorch_cos_sim(t_a, t_b)[0]# embeddings)
        else:
            similarities = []
        for similarity, kbent in zip(similarities, self.kb):
            if not e_type == kbent.e_type:
                continue
            # similarity = cosine_similarity(embedding, kbent.embedding)
            if similarity > max_score:
                best_match = kbent
                max_score = similarity
            if max_score > 0.95: # Very confident that this is the same, search no more
                break

        if max_score > 0.7:
            # Linked!
            return best_match.linked_name
        # Add new entity
        kbentity = KBEntity(text, e_type, "{}_{}".format(e_type, len(self.kb)), embedding)
        self.kb.append(kbentity)
        # self.save_kb_to_file(self.kb_file)
        return kbentity.linked_name
    


def do_bnslp(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logging.set_verbosity(logging.INFO) # ERROR INFO
    linker = Linker()
    detector = NERDetector("", "", "saved_model/")

    for subdir, dirs, files in os.walk(src_dir): 
        for filename in files:
            if not filename.endswith('.txt'):
                continue
            filepath = subdir + os.sep + filename
            # ann_filepath = out_dir  + os.sep + filename.replace(".txt",".out")
            # keep dir tree as well
            ann_filepath = filepath.replace(".txt", ".out")
            ann_filepath = ann_filepath.replace(src_dir, out_dir)

            Path(ann_filepath).parent.mkdir(parents=True, exist_ok=True) 

            document = Document(filepath)

            for line in document.segmented_text_lines:
                # returns a list of NamedEntity (context text label ..self.lemma)
                detected_entities = detector.get_entities(line)

                for entity in detected_entities:
                    ent_id = linker.get_id(entity.text, entity.label)
                    linked_entity = Entity(entity.text, entity.lemma, entity.label, ent_id)
                    # Document contains entities Entity = namedtuple('Entity', 'text lemma type linked_name')
                    document.entities.append(linked_entity)
            print("savign ann to {}".format(ann_filepath))
            document.save_as_out(ann_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BSNLP2021 file annotator')
    parser.add_argument('raw_doc_dir', help="Directory wiht files to annotate")
    parser.add_argument('annotation_dir', help="Output dir, will write *.out there")
    args = parser.parse_args()
    do_bnslp(args.raw_doc_dir, args.annotation_dir)
