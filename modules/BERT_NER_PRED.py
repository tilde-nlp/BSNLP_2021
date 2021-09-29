import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import argparse

import tensorflow as tf
from tensorflow.contrib import predictor
tf.get_logger().setLevel('ERROR')

from absl import flags,logging
from bert import modeling
from bert import optimization
from bert import tokenization

#import metrics
import numpy as np
import copy
from shutil import copyfile
from pathlib import Path
FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("label_file", None,
                    """The label file containing newline separated labels
                     of classes in data. Example: 'O'\n 'B-PER'\n etc.""")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")


flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("crf", True, "use crf!")



class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label

    # untokenized data
    self.tokens = text.split()
    self.labels = []
    if label: self.labels = label.split()

    # tokenized data
    self.tokenized_tokens = []
    self.tokenized_labels = []


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls,input_file):
        """Reads a BIO data"""
        with open(input_file, 'r', encoding='utf-8') as rf:
            lines = [];words = [];labels = []
            for line in rf:
                contents = line.strip()
                if contents.startswith("-DOCSTART-"):
                    continue
                if len(contents) == 0:  # newline
                    if len(words) == 0: continue
                    assert(len(words) == len(labels))
                    words_string = ' '.join(words)
                    labels_string = ' '.join(labels)
                    lines.append([words_string, labels_string])
                    words = []; labels = []
                    continue
                tokens = line.strip().split(' ')
                if (len(tokens) > 2):
                    print("more than 2 tokens in line _{}_".format(line.strip()))
                # assert(len(tokens) == 2) # Datos ir starpas
                word  = tokens[0]
                label = tokens[-1]
                if len(tokens) == 1:
                    label = "__"
                words.append(word)
                labels.append(label)
        return lines


class NerProcessor(DataProcessor):
    def __init__(self):
        self.labels = self.get_labels()

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_examples_from_text(self, text):
        lines = []
        text_lines = text.split('\n')
        for text_line in text_lines:
            line = []
            tokens = text_line.split(' ')
            words = []
            labels = []
            for token in tokens:
                token = token.strip()
                label = "__"
                words.append(token)
                labels.append(label)
            assert(len(words) == len(labels))
            words_string = ' '.join(words)
            labels_string = ' '.join(labels)
            lines.append([words_string, labels_string])
        return self._create_example(lines, "test")


    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        Load labels from file in data directory:
        """
        data_labels = []
        with open(os.path.join(FLAGS.label_file), 'r', encoding="utf-8") as labelfile:
            data_labels = [line.strip() for line in labelfile if line.strip()]
        all_labels = ["[PAD]"] + data_labels + ["X", "[CLS]", "[SEP]"]
        return all_labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[0])
            labels = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    
    # save tokens, poss, chunks, labels back to example
    example.tokenized_tokens = tokens
    example.tokenized_labels = labels

    ntokens = []
    segment_ids = []
    label_ids = []

    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        try:
            label_ids.append(label_map[labels[i]])
        except: # "__"
            label_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(label_map["[PAD]"])
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode=None):
    tf_features = []
    tf_examples = []
    for (ex_index, example) in enumerate(examples):
        feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_features.append(features)
        tf_examples.append(tf_example.SerializeToString())
    return tf_examples

def hidden2tag(hiddenlayer,numclass):
    linear = tf.keras.layers.Dense(numclass,activation=None)
    return linear(hiddenlayer)

def crf_loss(logits,labels,mask,num_labels,mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    #TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
                "transition",
                shape=[num_labels,num_labels],
                initializer=tf.contrib.layers.xavier_initializer()
        )
    
    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits,labels,transition_params =trans ,sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)
   
    return loss,transition

def softmax_layer(logits,labels,num_labels,mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask,dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict

def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config = bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
        )

    output_layer = model.get_sequence_output()
    #output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer,num_labels)
    # TODO test shape
    logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    if FLAGS.crf:
        mask2len = tf.reduce_sum(mask,axis=1)
        loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits,predict)

    else:
        loss,predict  = softmax_layer(logits, labels, num_labels, mask)

        return (loss, logits, predict)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.crf:
            (total_loss, logits,predicts) = create_model(bert_config, is_training, input_ids,
                                                            mask, segment_ids, label_ids,num_labels, 
                                                            use_one_hot_embeddings)

        else:
            (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                            mask, segment_ids, label_ids,num_labels, 
                                                            use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names=None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits,num_labels,mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels-1, weights=mask)
                return {
                    "confusion_matrix":cm
                }
                #
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


class NamedEntity():
    def __init__(self, text, label=None, context='', lemmas=[], offset_start=0):
        self.context = context
        self.text = text
        self.label = label
        self.offset_start = 0
        if self.offset_start == 0 and self.context:
            self.offset_start = self.context.find(self.text)
        self.offset_end = 0
        if self.offset_start > -1:
            self.offset_end = self.offset_start + len(self.text) 
        self.lemmas = lemmas 
        self.lemma = ''
        #self.get_lemma()

    def get_lemma(self):
        """Normalizācija"""
        if not self.lemmas:
            self.lemma = self.text
            return
        text_words = self.text.split()
        if not (len(text_words) == len(self.lemmas)):
            print("check tokenization: words {} lemmas {}".format(len(text_words), len(self.lemmas)))
        if self.label in ['PER', 'PERS']:
            # Person-like lemmatization - every word has to be lemmatized
            self.lemma = ' '.join(self.lemmas)
            # Strip punctuation from beginning and end
            self.lemma = self.lemma.strip(".!?") 
            # Handle bugs in stanza lemmatizer - Jāņa Ziediņa etc.
            for bad_lemma , good_lemma in NamedEntity.stanza_person_exceptions.items():
                self.lemma = self.lemma.replace(bad_lemma, good_lemma)

        # elif self.label in ['LOC','ORG','DATE', '']:
        else:
            # Locations like "paula stradiņa klīniskā universitātes slimnīca"
            # lemmatize only last word
            #@TODO vajag labāku pieeju. 
            # Varbūt atstāt tikai ģenitīvu, un citus locījumus (lokatīvu) normalizēt? 
            lemmas_to_join = text_words[:-1] + self.lemmas[-1:]
            self.lemma = ' '.join(lemmas_to_join)


class NERDetector():

    def __init__(self, model_dir=None, output_dir=None, saved_model_dir=None):

       
        FLAGS([
            './NERApp.py',
            '--do_lower_case=False',
            '--crf=True',
            '--do_predict=True',
            '--max_seq_length=128',
            '--train_batch_size=4',
            '--learning_rate=2e-5',
        ])

        if saved_model_dir and os.path.exists(saved_model_dir):
            FLAGS.vocab_file=os.path.join(saved_model_dir, 'vocab.txt')
            FLAGS.label_file=os.path.join(saved_model_dir, 'labels.txt')
        else:
            FLAGS.vocab_file=os.path.join(model_dir, 'vocab.txt')
            FLAGS.bert_config_file=os.path.join(model_dir, 'bert_config.json')
            FLAGS.init_checkpoint=os.path.join(model_dir, 'bert_model.ckpt')
            FLAGS.output_dir=output_dir
            FLAGS.label_file=os.path.join(output_dir, 'labels.txt')
            if not saved_model_dir:
                saved_model_dir="export"


        # self.nlp = stanza.Pipeline(lang=language, verbose=False, use_gpu=False) # Stanza pipeline


        self.processor = NerProcessor()
        self.label_list = self.processor.labels
        self.id2label = {key: value for key, value in enumerate(self.label_list)}
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        # Load from saved_model if present, else load from estimator and export to saved_model
        try:
            subdirs = [x for x in Path(saved_model_dir).iterdir() if x.is_dir() and'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
        except:
            bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
            if FLAGS.max_seq_length > bert_config.max_position_embeddings:
                raise ValueError(
                    "Cannot use sequence length %d because the BERT model "
                    "was only trained up to sequence length %d" %
                    (FLAGS.max_seq_length, bert_config.max_position_embeddings))

            tpu_cluster_resolver = None
            is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
            run_config = tf.contrib.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                master=FLAGS.master,
                model_dir=FLAGS.output_dir,
                save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host))
            train_examples = None
            num_train_steps = None
            num_warmup_steps = None
            
            model_fn = model_fn_builder(
                bert_config=bert_config,
                num_labels=len(self.label_list),
                init_checkpoint=FLAGS.init_checkpoint,
                learning_rate=FLAGS.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=FLAGS.use_tpu,
                use_one_hot_embeddings=FLAGS.use_tpu)
            self.estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=FLAGS.use_tpu,
                model_fn=model_fn,
                config=run_config,
                train_batch_size=FLAGS.train_batch_size,
                eval_batch_size=FLAGS.eval_batch_size,
                predict_batch_size=FLAGS.predict_batch_size,
                warm_start_from=FLAGS.output_dir)

            # https://stackoverflow.com/questions/56831583/saving-and-doing-inference-with-tensorflow-bert-model
            def serving_input_receiver_fn():
                seq_length=FLAGS.max_seq_length
                name_to_features = {
                    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
                    "mask": tf.FixedLenFeature([seq_length], tf.int64),
                    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
                    "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
                    }
                feature_spec = name_to_features
                serialized_tf_example = tf.placeholder(dtype=tf.string,
                                                        shape=[None],
                                                        name='input_example_tensor')
                receiver_tensors = {'example': serialized_tf_example}
                features = tf.parse_example(serialized_tf_example, feature_spec)
                return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

            # Export estimator as saved_model for future use
            self.estimator.export_saved_model(saved_model_dir, serving_input_receiver_fn)
            copyfile(FLAGS.vocab_file, os.path.join(saved_model_dir, 'vocab.txt'))
            copyfile(FLAGS.label_file, os.path.join(saved_model_dir, 'labels.txt'))
            # try to load saved model again now
            subdirs = [x for x in Path(saved_model_dir).iterdir() if x.is_dir() and'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
            # self.predict_fn = predictor.from_estimator(self.estimator, serving_input_receiver_fn=serving_input_receiver_fn)

        print("NER predictor init done.")


    def get_entities(self, input_text):
        start_time = time.time()
        if input_text == "":
            return []
            # return {"named entities": []}
        predict_examples = self.processor.get_examples_from_text(input_text)

        tf_feats = convert_examples_to_features(predict_examples, self.label_list,
                                                    FLAGS.max_seq_length, self.tokenizer)
        print("Example conversion done in {} seconds".format(time.time() - start_time))
        results = []
        for serialized_example in tf_feats:
            results.append(self.predict_fn({'example': [serialized_example]}))

        print("Prediction done in {} seconds".format(time.time() - start_time))
        predicted_examples = []
        for predict_example, prediction in zip(predict_examples, results):
            prediction = prediction["output"][0]

            predicted_example = copy.deepcopy(predict_example)
            predicted_example.labels = []

            tokens = predict_example.tokens
            labels = predict_example.labels
            
            tokenized_tokens = predict_example.tokenized_tokens
            tokenized_labels = predict_example.tokenized_labels
            text = predict_example.text
            length = len(tokenized_tokens)

            seq = 0
            last_label = "O"
            for token, label, p_id in zip(tokenized_tokens, tokenized_labels, prediction[1:length+1]):
                # print("token, label, p_label {}_{}_{}_".format(token, label, p_id))
                p_label = self.id2label[p_id]
                # print("token, label, p_label {}_{}_{}_".format(token, label, p_label))
                if label == 'X': continue
                if p_label == 'X': 
                    p_label = last_label
                if p_label in ['[CLS]','[SEP]']:
                    p_label = 'O'
                if 'I-' == p_label[0:2]:
                    if not p_label[2:] == last_label[2:]:
                        # Should not start with I-, replace with B-
                        p_label = 'B-'+ p_label[2:]
                last_label = p_label
                org_token = tokens[seq]
                # org_label = labels[seq]   # Oriģinālais labels __ neinteresē
                predicted_example.labels.append(p_label)
                seq += 1
            predicted_examples.append(predicted_example)

        print("predicted_examples list built in {} seconds".format(time.time() - start_time))

        # Extract entities from predicted_examples
        named_entities = []
        words_pos=[]
        for predicted_example in predicted_examples:
            # # Stanza is slow and uses different tokenization scheme.
            # doc = self.nlp(predicted_example.text)
            # for tsentence in doc.sentences:
            #     for tword in tsentence.words:
            #         words_pos.append(tword)

            current_entity_text = ''
            current_entity_label = ''
            current_entity_lemmas = []

            for token, label in zip(predicted_example.tokens,predicted_example.labels):
                # lemmas = []
                # while words_pos and words_pos[0].text in token:
                #     lemmas.append(words_pos.pop(0).lemma)
                #     nextw = words_pos[0] if 0 < len(words_pos) else None # Avoid AnnaAnna and JānisJānis
                #     if nextw and nextw.lemma in lemmas:
                #         break
                # try:
                #     lemmas = ''.join(lemmas)
                # except TypeError:
                #     lemmas = ""
                lemmas = ""
                # New entity starts - save current if present
                if current_entity_label and label[0] in ['B', 'O']:  
                    # Stanza lemmatizer bugs are handled in NamedEntity class
                    named_entity = NamedEntity(
                        current_entity_text, 
                        current_entity_label, 
                        context=predicted_example.text,
                        lemmas=current_entity_lemmas, 
                        offset_start=0)
                    named_entities.append(named_entity)
                    current_entity_text = ''
                    current_entity_label = ''
                    current_entity_lemmas = []
                # Skip non-entities
                if label == 'O': continue
                # Append text if entity
                current_entity_text += ' '+token
                current_entity_label = label[2:]
                current_entity_lemmas.append(lemmas)
            # if text did end with named entity, append it
            if current_entity_text and current_entity_label and current_entity_lemmas:
                named_entity = NamedEntity(
                    current_entity_text,
                    current_entity_label,
                    context=predicted_example.text,
                    lemmas=current_entity_lemmas,
                    offset_start=0)
                named_entities.append(named_entity)

            
        print("NE normalization done in {} seconds".format(time.time() - start_time))

        return named_entities
        # return {"named entities": [{
        #     "text": named_entity.text,
        #     "label": named_entity.label,
        #     "lemma": named_entity.lemma,
        #     } for named_entity in named_entities
        # ]}
        


def main(model_dir, output_dir, instring, saved_model_dir):
    logging.set_verbosity(logging.INFO) # ERROR INFO

    detector = NERDetector(model_dir, output_dir, saved_model_dir)
    detected_entities = detector.get_entities(instring)
    print("Detected entities:\n{}".format(detected_entities))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, help="directory containing model files")
    parser.add_argument("--output_dir", default=None, help="directory containing fine-tuned model files")
    parser.add_argument("--instring", help="text, in which to detect entities")
    parser.add_argument("--saved_model_dir", help="dir containing saved model, bert config and labels files")

    args = parser.parse_args()
    print("Starting:")
    print(f"model_dir: {args.model_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"saved_model_dir: {args.saved_model_dir}")
    print(f"instring: {args.instring}")


    main(args.model_dir, args.output_dir, args.instring, args.saved_model_dir)
