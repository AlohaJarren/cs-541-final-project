# final_project.py
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
            if label in {"dog","cat"}:
                data.append((label, text))
    return data

# preprocess
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
SPACE_RE = re.compile(r"\s+")


# stop words we want to remove
STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he','in','is',
    'it','its','of','on','that','the','to','was','will','with','the','this',
    'but','they','have','had','what','when','where','who','which','why','how',
    'all','would','there','their','been','if','out','so','up','more','her','she',
    'him','very','can','our','we'
}


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
def stratified_split(items, test_size=0.6, seed=42):
    rnd = random.Random(seed)
    dogs = [x for x in items if x[0] == "dog"]
    cats = [x for x in items if x[0] == "cat"]
    rnd = random.Random(seed)
    rnd.shuffle(dogs)
    rnd.shuffle(cats)
    nd = int(len(dogs) * (1 - test_size))
    nc = int(len(cats) * (1 - test_size))
    train = dogs[:nd] + cats[:nc]
    test  = dogs[nd:] + cats[nc:]
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


# Naive Bayes model (multinomial, Laplace)
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
        V = len(self.vocab)
        for c in self.class_counts:
            self.log_prior[c] = math.log(self.class_counts[c] / max(n, 1))
            total = sum(self.token_counts[c].values())
            self.denom[c] = total + self.alpha * V

    def predict_one(self, toks):
        # sum logs of P(c) + sum log P(w|c)
        scores = {}
        V = len(self.vocab)
        for c in self.class_counts:
            s = self.log_prior[c]
            for w in toks:
                if w in self.vocab:
                    num = self.token_counts[c][w] + self.alpha
                    s += math.log(num / self.denom[c])
                else:
                    s += math.log(self.alpha / self.denom[c]) if V > 0 else 0.0
            scores[c] = s
        return max(scores, key=scores.get)


# quick report
def evaluate(y_true, y_pred):
    labels = ["dog","cat"]
    acc = (sum(1 for a,b in zip(y_true,y_pred) if a==b) / len(y_true)) if y_true else 0.0
    cm = {a:{b:0 for b in labels} for a in labels}
    for t,p in zip(y_true, y_pred):
        cm[t][p] += 1
    return acc, cm

# create informative tokens
def informative_words(nb, k=8):
    scores = []
    for w in nb.vocab:
        pd = (nb.token_counts["dog"][w] + nb.alpha) / nb.denom["dog"]
        pc = (nb.token_counts["cat"][w] + nb.alpha) / nb.denom["cat"]
        scores.append((math.log(pd) - math.log(pc), w))
        
    scores.sort()
    cat_top = [w for _, w in scores[:k]]
    dog_top = [w for _, w in scores[-k:]]
    return dog_top, cat_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="pets.txt", help="input data file")
    ap.add_argument("--test_size", type=float, default=0.2, help="test split ratio")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    ap.add_argument("--keep_stop", action="store_true", help="do NOT remove stopwords")
    ap.add_argument("--samples", type=int, default=5, help="show N sample predictions")
    args = ap.parse_args()

    data = load_data(args.file)
    print(f"Loading {len(data)} pet descriptions...")
    if not data:
        print("No data found. Did you place pets.txt in the same folder?")
        return

    dogs = sum(1 for lbl,_ in data if lbl=="dog")
    cats = sum(1 for lbl,_ in data if lbl=="cat")
    print(f"Total Dogs: {dogs}, Total Cats: {cats}\n")

    proc = [(lbl, preprocess(txt, use_stop=not args.keep_stop)) for lbl, txt in data]
    train, test = stratified_split(proc, test_size=args.test_size, seed=args.seed)
    print(f"Training: {len(train)} samples")
    print(f"Testing:  {len(test)} samples\n")

    print("Training Naive Bayes...")
    nb = NaiveBayes(alpha=args.alpha)
    nb.fit(train)
    print(f"Vocabulary size: {len(nb.vocab)} words\n")

    y_true = [lbl for lbl,_ in test]
    y_pred = [nb.predict_one(toks) for _, toks in test]
    acc, cm = evaluate(y_true, y_pred)

    print(f"Accuracy: {acc:.2%}\n")
    print("Confusion Matrix:")
    print("         Pred:Dog  Pred:Cat")
    print(f"True:Dog    {cm['dog']['dog']:3}       {cm['dog']['cat']:3}")
    print(f"True:Cat    {cm['cat']['dog']:3}       {cm['cat']['cat']:3}\n")

    dog_tp = cm['dog']['dog']; dog_fp = cm['cat']['dog']; dog_fn = cm['dog']['cat']
    cat_tp = cm['cat']['cat']; cat_fp = cm['dog']['cat']; cat_fn = cm['cat']['dog']
    dog_prec = dog_tp / (dog_tp + dog_fp) if (dog_tp + dog_fp) else 0.0
    dog_rec  = dog_tp / (dog_tp + dog_fn) if (dog_tp + dog_fn) else 0.0
    cat_prec = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) else 0.0
    cat_rec  = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) else 0.0

    print("Performance Metrics:")
    print(f"Dog - Precision: {dog_prec:.2%}, Recall: {dog_rec:.2%}")
    print(f"Cat - Precision: {cat_prec:.2%}, Recall: {cat_rec:.2%}\n")

    print("Most important keywords for classification:")
    dtop, ctop = informative_words(nb, k=10)
    print("Top 10 dog keywords:")
    for w in dtop:
        print(f"  {w}")
    print("\nTop 10 cat keywords:")
    for w in ctop:
        print(f"  {w}")
    print()

    # sample predictions
    rnd = random.Random(args.seed)
    show = min(args.samples, len(test))
    if show:
        print("Some prediction examples:")
        print("-" * 50)
        for lbl, toks in rnd.sample(test, k=show):
            guess = nb.predict_one(toks)
            txt = " ".join(toks[:20])
            print(f"Text: \"{txt}...\"")
            print(f"Actual: {lbl}, Predicted: {guess}")
            print("Result:", "CORRECT!" if lbl == guess else "WRONG")
            print()

if __name__ == "__main__":
    main()