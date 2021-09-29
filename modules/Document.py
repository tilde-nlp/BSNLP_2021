# coding='utf-8'
"""
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
"""
from collections import namedtuple
import nltk
import regex as re
Entity = namedtuple('Entity', 'text lemma type linked_name')
import stanza

class Document():
    def __init__(self, text_file):
        self.text_file = text_file
        self.doc_id = ''
        self.doc_language = ''
        self.doc_creation_date = ''
        self.doc_url = ''
        self.doc_title = ''
        self.original_text = ''
        self.segmented_text_lines = []
        self.entities = []

        self.load_txt_and_segment(self.text_file)


    def load_entities_from_ann(self, ann_file):
        if self.entities:
            # should not load if already loaded
            return
        with open(ann_file, 'r', encoding='utf-8') as annfile:
            for i, line in enumerate(annfile.readlines()):
                if i == 0:
                    assert self.doc_id == line.strip()
                    continue
                line = line.strip()
                elems = line.split('\t')
                if not len(elems) == 4:
                    raise Exception('Wrong number of elems in file {}'.format(ann_file))
                    # continue # first line or error
                entity = Entity(elems[0], elems[1], elems[2], elems[3])
                self.entities.append(entity)


    def load_entities_from_conll(self, conll):
        pass


    def load_txt_and_segment(self, txt_file):
        # <ID><LANGUAGE><CREATION-DATE><URL><TITLE>
        with open(txt_file, 'r', encoding='utf-8') as txtfile:
            for i, line in enumerate(txtfile.readlines()):
                if i == 0:
                    self.doc_id = line.strip()
                elif i == 1:
                    self.doc_language = line.strip()
                elif i == 2:
                    self.doc_creation_date = line.strip()
                elif i == 3:
                    self.doc_url = line.strip()
                elif i == 4:
                    self.doc_title = line.strip()
                else:
                    self.original_text = self.original_text + line
        if self.doc_title.strip():
            self.segmented_text_lines.append(self.doc_title.strip())
        def segment_text_to_sentences(text):
            # Does not have to be perfect but just to split enough to process with bert

            sentences = []
            lines = text.split('\n')
            for line in lines:
                # https://github.com/Mottl/ru_punkt
                languages = {
                    'bg':'russian', # At this moment no segmenter, use simplest tokenizer - Russian
                    'cs':'czech',
                    'pl':'polish',
                    'ru':'russian',
                    'sl':'slovene',
                    'uk':'russian', # At this moment no segmenter, use simplest tokenizer - Russian
                }
                try:
                    language = languages[self.doc_language]
                except:
                    language = 'russian'
                sents = nltk.sent_tokenize(line, language=language)

                sentences = sentences + sents

            return sentences

        self.segmented_text_lines = self.segmented_text_lines + segment_text_to_sentences(self.original_text)
        tokenized_sents = [] # More like bert tokenization
        for sentence in self.segmented_text_lines:
            # sentence = re.sub(r"(?<=[\p{L}\'\"])(?<![\p{Lu},\.\!\?\:\-\'\"\s]\p{Ll})(?!=[,\.\!\?\:\-][\p{L}\d])(?=[,\.\!\?\:\-])"," ",sentence)
            sentence = sentence.replace('‍', '') # Zero width space 
            sentence = sentence.replace('"', ' " ')
            sentence = sentence.replace('»', ' » ')
            sentence = sentence.replace('«', ' « ')
            sentence = sentence.replace('„', ' „ ')
            sentence = sentence.replace('”', ' ” ')
            sentence = sentence.replace('“', ' “ ')
            sentence = sentence.replace('(', ' ( ')
            sentence = sentence.replace(')', ' ) ')

            sentence = re.sub(r"(?<=\p{L}\p{L}[\p{L}\'\"])(?=[\,\.\!\?\:\-]\s)"," ",sentence)
            sentence = re.sub(r"(?<=[\p{L}\'\"])(?=[\,\.\!\?\:\-]$)"," ",sentence)
            sentence = re.sub(r"(?<=[\p{L}\'\"])(?=,\s)"," ",sentence)

            sentence = re.sub(r"^\s","",sentence)
            sentence = re.sub(r"\s$","",sentence)
            sentence = re.sub(r" +"," ",sentence)
            tokenized_sents.append(sentence)
        self.segmented_text_lines = tokenized_sents


    def save_as_conll(self, conll_file):
        # Reorder entities longest first to enable correct saving
        self.entities.sort(key=lambda x: len(x.text), reverse=True)

        # with open("dump.txt", 'w', encoding='utf-8')as dumpf:
        #     for ent in self.entities:
        #         dumpf.write("{}\n".format(ent.text))
        #     dumpf.write("\n\n")
        #     dumpf.write("\n".join(self.segmented_text_lines))
        #     dumpf.write("\n\n")

        IOBstring = ""
        for sentence in self.segmented_text_lines:
            def parse_sent(sent):
                words = sent.split(" ")

                resstring = ""
                resstring = " O\n".join(words)
                resstring = resstring + " O\n"

                for entity in self.entities:
                    entity_words = entity.text.split(" ")

                    before_ent_string = ""
                    before_ent_string = " O\n".join(entity_words)
                    before_ent_string = before_ent_string + " O\n"

                    after_entity_string = ""
                    for j, e_word in enumerate(entity_words):
                        if j == 0:
                            after_entity_string = after_entity_string + e_word + " " + "B-" + entity.type + "\n"
                            continue
                        after_entity_string = after_entity_string + e_word + " " + "I-" + entity.type + "\n"
                    resstring = resstring.replace(before_ent_string, after_entity_string)
                return resstring

            IOBstring = IOBstring + parse_sent(sentence) + "\n"
        with open(conll_file, 'w', encoding='utf-8') as conllf:
            conllf.write(IOBstring)
        return IOBstring


    def save_as_out(self, out_file):
        # Reorder entities alphabetically
        self.entities.sort(key=lambda x: x.text, reverse=False)
        # Must load detector for each document with appropriate language 
        # Documents followning each other are often in the same lang
        self.nlp = None # stanza.Pipeline(lang=self.doc_language, verbose=False, use_gpu=False) # Stanza pipeline


        def ent_in_list(ent, list_of_ents):
            for ent_of_list in list_of_ents:
                if ent_of_list.text.lower() == ent.text.lower():
                    # lower() because " the system response should include only one annotation for all of these mentions."
                    return True
            return False

        with open(out_file, 'w', encoding='utf-8') as annfile:
            # first line - file id:
            annfile.write("{}\n".format(self.doc_id))
            
            outents = []
            for entity in self.entities:
                # Write out only unique entities
                if ent_in_list(entity, outents):
                    continue
                outents.append(entity)

            for entity in outents:
                if not entity.lemma:
                    if not self.nlp: # Set up nlp if not already done
                        self.nlp = stanza.Pipeline(lang=self.doc_language, verbose=False)#, use_gpu=False
                    doc = self.nlp(entity.text.strip())
                    try:
                        e_lemma = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
                    except:
                        e_lemma = entity.text.strip()
                lemma = e_lemma if e_lemma else entity.text.strip()
                #lemma = entity.text.strip()
                annfile.write("{}\t{}\t{}\t{}\n".format(entity.text.strip(), lemma, entity.type, entity.linked_name))

