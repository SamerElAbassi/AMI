# coding: utf8
from __future__ import unicode_literals

from spacy.symbols import NOUN, PROPN, PRON, VERB, AUX
from spacy.errors import Errors
from spacy.util import filter_spans

def noun_chunks_as_spans(doc):
    return [doc[start:end] for start,end,_ in noun_chunks(doc)]

def noun_chunks(doclike):
    doc = doclike.doc

    if not doc.is_parsed:
        raise ValueError(Errors.E029)

    if not len(doc):
        return
    np_label = doc.vocab.strings.add("NP")
    left_labels = ["det", "fixed", "neg"]  # ['nunmod', 'det', 'appos', 'fixed']
    right_labels = ["flat", "fixed", "compound", "neg"]
    stop_labels = ["punct"]
    np_left_deps = [doc.vocab.strings.add(label) for label in left_labels]
    np_right_deps = [doc.vocab.strings.add(label) for label in right_labels]
    stop_deps = [doc.vocab.strings.add(label) for label in stop_labels]
    token = doc[0]
    while token and token.i < len(doclike):
        if token.pos in [PROPN, NOUN, PRON]:
            left, right = noun_bounds(
                doc, token, np_left_deps, np_right_deps, stop_deps
            )
            yield left.i, right.i + 1, np_label
            token = right
        token = next_token(token)


def is_verb_token(token):
    return token.pos in [VERB, AUX]


def next_token(token):
    try:
        return token.nbor()
    except IndexError:
        return None


def noun_bounds(doc, root, np_left_deps, np_right_deps, stop_deps):
    left_bound = root
    for token in reversed(list(root.lefts)):
        if token.dep in np_left_deps:
            left_bound = token
    right_bound = root
    for token in root.rights:
        if token.dep in np_right_deps:
            left, right = noun_bounds(
                doc, token, np_left_deps, np_right_deps, stop_deps
            )
            if list(
                filter(
                    lambda t: is_verb_token(t) or t.dep in stop_deps,
                    doc[left_bound.i : right.i],
                )
            ):
                break
            else:
                right_bound = right
    return left_bound, right_bound


def find_all_candidates(doc, noun_chunk_spans):
    """Looks for all the candidates including overlapping ones.
    """
    all_nounps = []
    already_in = set()
    for noun_chunk in noun_chunk_spans:
        all_nounps.append(noun_chunk)
        already_in.add(noun_chunk.text.strip())
        # extract phrase from left to right edges of the noun chunk
        noun_phrase = doc.doc[noun_chunk.root.left_edge.i:noun_chunk.root.right_edge.i+1]
        if not noun_phrase.text.strip() in already_in:
            all_nounps.append(noun_phrase)
            already_in.add(noun_phrase.text.strip())
    return filter_spans(all_nounps)


def noun_phrases(doclike):
    doc = doclike.doc
    noun_chunk_spans = noun_chunks_as_spans(doc)
    nps = find_all_candidates(doc, noun_chunk_spans)
    return nps
