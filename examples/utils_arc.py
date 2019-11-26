# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load ARC dataset. """

from __future__ import absolute_import, division, print_function

import json
import json_lines
import logging
import math
import collections
import getpass
from io import open
from tqdm import tqdm

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)


class ArcExample(object):
    """
    A single training/test example for the ARC dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 context_tokens,
                 context_text,
                 answer_text=None,
                 correct=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_tokens = context_tokens
        self.context_text = context_text
        self.answer_text = answer_text
        self.correct = correct
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.context_tokens))
        s += ", answer_text: %s" % (self.answer_text)
        s += ", correct: %r" % (self.correct)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 correct,
                 qas_id,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.correct = correct
        self.qas_id = qas_id
        self.is_impossible = is_impossible

class GroupedFeatures(object):

    def __init__(self,
                 unique_ids,
                 example_indices,
                 doc_span_indices,
                 tokens_,
                 token_to_orig_maps_,
                 input_ids_,
                 input_masks,
                 segment_ids_,
                 cls_indices,
                 p_masks,
                 paragraph_lens,
                 label,
                 qas_ids):
        self.unique_ids = unique_ids
        self.example_indices = example_indices
        self.doc_span_indices = doc_span_indices
        self.tokens_ = tokens_
        self.token_to_orig_maps_ = token_to_orig_maps_
        self.input_ids_ = input_ids_
        self.input_masks = input_masks
        self.segment_ids_ = segment_ids_
        self.cls_indices = cls_indices
        self.p_masks = p_masks
        self.paragraph_lens = paragraph_lens
        self.label = label
        self.qas_ids = qas_ids


def read_arc_examples(input_file, is_training, version_2_with_negative):
    """Read a ARC jsonl file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        # input_data = json.load(reader)["data"]
        input_data = []
        reader = json_lines.reader(reader)
        for line in reader:
            input_data.append(line)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        q_id = entry['id']
        question_text = entry['question']['stem']

        for info in entry['question']['choices']:
            context_text = info['para']
            doc_tokens = []
            prev_is_whitespace = True
            for c in context_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False

            answer_label = info['label']
            qas_id = q_id + '*-*' + answer_label
            answer_text = info['text']
            is_impossible = False
            correct = None
            if is_training:
                if not is_impossible:
                    correct = answer_label == entry['answerKey']

            example = ArcExample(
                qas_id=qas_id,
                question_text=question_text,
                context_tokens=doc_tokens,
                context_text=context_text,
                answer_text=answer_text,
                correct=correct,
                is_impossible=is_impossible)
            examples.append(example)
        if len(examples) >= 20:
            break

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)
    max_answer_query_length = max_seq_length

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):
        assert isinstance(example, ArcExample)

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)
        answer_tokens = tokenizer.tokenize(example.answer_text)

        answer_query_tokens = query_tokens + answer_tokens

        if len(answer_query_tokens) > max_answer_query_length:
            answer_query_tokens = answer_query_tokens[0:max_answer_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.context_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # tok_start_position = None
        # tok_end_position = None
        # if is_training and example.is_impossible:
        #     tok_start_position = -1
        #     tok_end_position = -1
        # if is_training and not example.is_impossible:
        #     tok_start_position = orig_to_tok_index[example.start_position]
        #     if example.end_position < len(example.doc_tokens) - 1:
        #         tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        #     else:
        #         tok_end_position = len(all_doc_tokens) - 1
        #     (tok_start_position, tok_end_position) = _improve_answer_span(
        #         all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
        #         example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(answer_query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        # doc_spans = []
        # start_offset = 0
        # while start_offset < len(all_doc_tokens):
        #     length = len(all_doc_tokens) - start_offset
        #     if length > max_tokens_for_doc:
        #         length = max_tokens_for_doc
        #     doc_spans.append(_DocSpan(start=start_offset, length=length))
        #     if start_offset + length == len(all_doc_tokens):
        #         break
        #     start_offset += min(length, doc_stride)

        doc_spans = [_DocSpan(start=0, length=len(all_doc_tokens))]

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            # token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += answer_query_tokens
                segment_ids += [sequence_a_segment_id] * len(answer_query_tokens)
                p_mask += [1] * len(answer_query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                # is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                #                                        split_token_index)
                # token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                tokens += answer_query_tokens
                segment_ids += [sequence_b_segment_id] * len(answer_query_tokens)
                p_mask += [1] * len(answer_query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            # start_position = None
            # end_position = None
            # if is_training and not span_is_impossible:
            #     # For training, if our document chunk does not contain an annotation
            #     # we throw it out, since there is nothing to predict.
            #     doc_start = doc_span.start
            #     doc_end = doc_span.start + doc_span.length - 1
            #     out_of_span = False
            #     if not (tok_start_position >= doc_start and
            #             tok_end_position <= doc_end):
            #         out_of_span = True
            #     if out_of_span:
            #         start_position = 0
            #         end_position = 0
            #         span_is_impossible = True
            #     else:
            #         if sequence_a_is_doc:
            #             doc_offset = 0
            #         else:
            #             doc_offset = len(query_tokens) + 2
            #         start_position = tok_start_position - doc_start + doc_offset
            #         end_position = tok_end_position - doc_start + doc_offset

            # if is_training and span_is_impossible:
            #     start_position = cls_index
            #     end_position = cls_index

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                # logger.info("token_is_max_context: %s" % " ".join([
                #     "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                # ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                # if is_training and not span_is_impossible:
                    # answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    # logger.info("start_position: %d" % (start_position))
                    # logger.info("end_position: %d" % (end_position))
                    # logger.info(
                    #     "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    # token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    # start_position=start_position,
                    # end_position=end_position,
                    is_impossible=span_is_impossible,
                    correct=example.correct,
                    qas_id=example.qas_id))
            unique_id += 1
        if len(features) >= 20:
            break

    return features

def features_to_groups(features):
    groups = []
    saved_qid = ''
    saved_features = []

    for f in features:
        assert isinstance(f, InputFeatures)
        current_qid = f.qas_id[:f.qas_id.index('*-*')]

        if current_qid == saved_qid:
            saved_features.append(f)
        else:
            if len(saved_features) == 4:
                groups.append(GroupedFeatures(unique_ids=[f.unique_id for f in saved_features],
                                              example_indices=[f.example_index for f in saved_features],
                                              doc_span_indices=[f.doc_span_index for f in saved_features],
                                              tokens_=[f.tokens for f in saved_features],
                                              token_to_orig_maps_=[f.token_to_orig_map for f in saved_features],
                                              input_ids_=[f.input_ids for f in saved_features],
                                              input_masks=[f.input_mask for f in saved_features],
                                              segment_ids_=[f.segment_ids for f in saved_features],
                                              cls_indices=[f.cls_index for f in saved_features],
                                              p_masks=[f.p_mask for f in saved_features],
                                              paragraph_lens=[f.paragraph_len for f in saved_features],
                                              label=[f.correct for f in saved_features].index(True),
                                              qas_ids=[f.qas_id for f in saved_features]))
            saved_qid = current_qid
            saved_features = [f]

    if len(saved_features) == 4:
        groups.append(GroupedFeatures(unique_ids=[f.unique_id for f in saved_features],
                                      example_indices=[f.example_index for f in saved_features],
                                      doc_span_indices=[f.doc_span_index for f in saved_features],
                                      tokens_=[f.tokens for f in saved_features],
                                      token_to_orig_maps_=[f.token_to_orig_map for f in saved_features],
                                      input_ids_=[f.input_ids for f in saved_features],
                                      input_masks=[f.input_mask for f in saved_features],
                                      segment_ids_=[f.segment_ids for f in saved_features],
                                      cls_indices=[f.cls_index for f in saved_features],
                                      p_masks=[f.p_mask for f in saved_features],
                                      paragraph_lens=[f.paragraph_len for f in saved_features],
                                      label=[f.correct for f in saved_features].index(True),
                                      qas_ids=[f.qas_id for f in saved_features]))

    #TODO make this better...

    return groups


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "answer_logits"])


def write_predictions(all_examples, all_features, all_results,
                      output_prediction_file, verbose_logging):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))

    qas_to_examples = {}
    for example in all_examples:
        assert isinstance(example, ArcExample)
        qas_to_examples[example.qas_id] = example

    ui_to_results = {}
    for result in all_results:
        assert isinstance(result, RawResult)
        ui_to_results[result.unique_id] = result

    ui_to_features = {}
    for grouped_feature in all_features:
        assert isinstance(grouped_feature, GroupedFeatures)
        ui_to_features['*'.join(grouped_feature.unique_ids)] = grouped_feature

    all_predictions = collections.OrderedDict()

    with open(output_prediction_file, 'w') as writer:
        for gf in all_features:
            assert isinstance(gf, GroupedFeatures)
            current_unique_ids = gf.unique_ids
            current_unique_id = '*'.join(current_unique_ids)
            current_qa_ids = gf.qas_ids

            current_result = ui_to_results[current_unique_id]
            _, max_index = current_result.max()

            for i, qa in enumerate(current_qa_ids):
                current_example = qas_to_examples[qa]

                assert isinstance(current_example, ArcExample)

                if i is 0:

                    current_id = current_example.qas_id[:current_example.qas_id.index('*-*')]
                    writer.write(current_id + '\n')
                    current_question = current_example.question_text
                    writer.write(current_question + '\n\n')

                    all_predictions[current_id] = 0

                selected = '*' if max_index == i else ''
                correct = '+' if current_example.correct else ''

                if bool(selected) and bool(correct):
                    all_predictions[current_id] = 1

                writer.write('{}{}{} {}\n'.format(current_result[i], selected, correct, current_example.answer_text))

            writer.write('\n')

        writer.close()

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


if __name__ == '__main__':
    input_file = 'C://Users/Mitch/PycharmProjects/ARC/ARC-with-context/dev.jsonl'
    evaluate = False
    version_2_with_negative = False

    examples = read_arc_examples(input_file=input_file,
                                 is_training=not evaluate,
                                 version_2_with_negative=version_2_with_negative)

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True,
                                              cache_dir=None)

    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=512,
                                            is_training=not evaluate,
                                            cls_token_segment_id=0,
                                            pad_token_segment_id=0,
                                            cls_token_at_end=False,
                                            sequence_a_is_doc=False)

    # this seems to work 11/22, but
    # TODO still update the write predictions and evaluate since don't know what the results look like yet

    grouped_features = features_to_groups(features)

    print('hi')