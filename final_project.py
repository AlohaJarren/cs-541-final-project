'''
Naive Bayes Pet Classifier (Dog vs Cat)
CS 441/541 Final Group Project
Jarren Calizo & Vivi Chen
'''

import math
import random
import re
import argparse
from collections import Counter, defaultdict

# load data from pets.txt
def load_data(filename="pets.txt"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line:
                continue
            label, text = line.split(" ", 1)
            label = label.lower()
            if label in {"dog", "cat"}:
                data.append((label, text))
    return data

# tokenize + clean
PUNCT_RE = re.compile('[^a-z0-9 ]+')
SPACE_RE = re.compile(' +')
STOPWORDS = set("a an and are as at be by for from has he in is it its of on that the to was were will with she his her".split())

def tokenize(text):
    t = text.lower()
    t = PUNCT_RE.sub(" ", t)
    t = SPACE_RE.sub(" ", t).strip()
    return [] if not t else t.split()

def preprocess(text, use_stop=True):
    toks = tokenize(text)
    if use_stop:
        toks = [w for w in toks if w not in STOPWORDS]
    return toks

# split preserving class ratio
def stratified_split(items, test_size=0.25, seed=17):
    dogs = [x for x in items if x[0] == "dog"]
    cats = [x for x in items if x[0] == "cat"]
    rnd = random.Random(seed)
    rnd.shuffle(dogs)
    rnd.shuffle(cats)
    nd = int(len(dogs) * (1 - test_size))
    nc = int(len(cats) * (1 - test_size))
    train = dogs[:nd] + cats[:nc]
    test = dogs[nd:] + cats[nc:]
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test

# NB model (multinomial, laplace)
class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = Counter()
        self.token_counts = defaultdict(Counter)
        self.vocab = set()
        self.log_prior = {}
        self.denom = {}

    def fit(self, labeled_tokens):
        for label, toks in labeled_tokens:
            self.class_counts[label] += 1
            for w in toks:
                self.token_counts[label][w] += 1
                self.vocab.add(w)
        n = sum(self.class_counts.values())
        for c in self.class_counts:
            self.log_prior[c] = math.log(self.class_counts[c] / n)
            total = sum(self.token_counts[c][w] for w in self.vocab)
            self.denom[c] = total + self.alpha * len(self.vocab)

    def predict_one(self, toks):
        scores = {}
        for c in self.class_counts:
            s = self.log_prior[c]
            for w in toks:
                if w in self.vocab:
                    s += math.log((self.token_counts[c][w] + self.alpha) / self.denom[c])
            scores[c] = s
        return max(scores, key=scores.get)

# quick report
def evaluate(y_true, y_pred):
    labels = ["dog", "cat"]
    acc = (sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)) if y_true else 0.0
    cm = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    lines = [f"accuracy: {acc:.3f}", "confusion:"]
    lines.append(f"    dog -> dog={cm['dog']['dog']} cat={cm['dog']['cat']}")
    lines.append(f"    cat -> dog={cm['cat']['dog']} cat={cm['cat']['cat']}")
    return ".join(lines)"

# create informative tokens
def informative_words(nb, k=8):
    scores = []
    for w in nb.vocab:
        pd = (nb.token_counts['dog'][w] + nb.alpha) / nb.denom['dog']
        pc = (nb.token_counts['cat'][w] + nb.alpha) / nb.denom['cat']
        scores.append((math.log(pd) - math.log(pc), w))
    scores.sort()
    dog_top = [w for _, w in scores[-k:]]
    cat_top = [w for _, w in scores[:k]]
    return dog_top, cat_top

def run(data, args):
    proc = [(lbl, preprocess(txt, use_stop=not args.keep_stop)) for lbl, txt in data]
    train, test = stratified_split(proc, test_size=args.test_size, seed=args.seed)
    nb = NaiveBayes(alpha=args.alpha)
    nb.fit(train)
    y_true = [lbl for lbl, _ in test]
    y_pred = [nb.predict_one(t) for _, t in test]
    print(evaluate(y_true, y_pred))
    dtop, ctop = informative_words(nb, k=8)
    print("top dog words:", dtop)
    print("top cat words:", ctop)
    rnd = random.Random(args.seed)
    print("samples:")
    for lbl, toks in rnd.sample(test, k=min(args.samples, len(test))):
        guess = nb.predict_one(toks)
        print(f"true={lbl} pred={guess} text={' '.join(toks[:20])}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="pets.txt")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--keep_stop", action="store_true")
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()
    data = load_data(args.file)
    print(f"Loaded {len(data)} pet descriptions.")
    if data:
        run(data, args)

if __name__ == "__main__":
    main()