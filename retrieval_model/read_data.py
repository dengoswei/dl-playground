# coding=utf-8
#

import json
import collections
import random


def load_word_embedding(filename, max_vocab_size):
    word2vec = {}
    word_counter = collections.Counter()
    with open(filename, 'r') as fin:
        raw = json.load(fin)
        word2vec = raw['word2vec']
        word_counter = raw['word_counter']

    word2idx = {}
    counter = collections.Counter()
    for w, t in word_counter.items():
        counter[w] = t
    # -NUL-: 0
    # -UNK-: 1
    for w, t in counter.most_common(max_vocab_size-1):
        word2idx[w] = len(word2idx) + 2
    return word2vec, word2idx

def load_sougou_data(filename, word2idx):
    q = []
    p = []
    l = []
    with open(filename, 'r') as fin:
        raw = json.load(fin)
        q = raw['q']
        p = raw['p']
        l = raw['l']
        assert len(q) == len(p)
        assert len(q) == len(l)

        def to_seqences(text):
            seq = []
            for w in text:
                if w in word2idx:
                    seq.append(word2idx[w])
                else:
                    seq.append(1)
            return seq
        q = map(to_seqences, q)
        p = map(to_seqences, p)
    return q, p, l

def filter_sougou_data(q, p, l):
    assert len(q) == len(p)
    assert len(q) == len(l)
    nq = []
    np = []
    nl = []
    for i, label in enumerate(l):
        if 1 == label:
            continue

        nq.append(q[i])
        np.append(p[i])
        nl.append(l[i])
    return nq, np, nl


def split_data(q, p, l, ratio):
    # split into train && validate
    validate_size = int(len(q) * ratio)
    assert 0 < validate_size
    assert validate_size < len(q)
    random_idx = random.sample(range(len(q)), len(q))
    new_q = [q[idx] for idx in random_idx]
    new_p = [p[idx] for idx in random_idx]
    new_l = [l[idx] for idx in random_idx]
    return new_q[:validate_size], \
            new_p[:validate_size], new_l[:validate_size], \
            new_q[validate_size:], new_p[validate_size:], \
            new_l[validate_size:]


def get_mini_batches(q, p, l, num_step, min_batch_size):
    assert len(q) == len(p)
    assert len(q) == len(l)
    align_len = len(q) - (len(q) % min_batch_size)
    q = q[:align_len]
    p = p[:align_len]
    l = l[:align_len]
    assert 0 == (len(q) % min_batch_size)
    idx = range(len(q))
    assert len(idx) == len(q)
    for _ in range(num_step):
        random_idx = random.sample(idx, len(idx))
        for i in range(0, len(random_idx), min_batch_size):
            yield q[i:i+min_batch_size], p[i:i+min_batch_size], l[i:i+min_batch_size]

