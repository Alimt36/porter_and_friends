
import argparse
import random
import re
import time
from collections import defaultdict

from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

try:
    import Stemmer as PyStemmer  # C implementation of Snowball
except ImportError:
    PyStemmer = None

# ----------------------------------------------------------------------------
# The stemmers we compare
# ----------------------------------------------------------------------------
porter_orig = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
porter_martin = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
porter_nltk = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)
snowball = SnowballStemmer("english")  # = Porter2
lancaster = LancasterStemmer()

STEMMERS = {
    "Porter(orig)": porter_orig.stem,
    "Porter(Martin)": porter_martin.stem,
    "Porter(NLTK)": porter_nltk.stem,
    "Snowball": snowball.stem,
    "Lancaster": lancaster.stem,
}

PAUSE = True


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def wait():
    if PAUSE:
        input("\n[press Enter for the next demo] ")


# ----------------------------------------------------------------------------
# My own implementation of the ORIGINAL Porter algorithm (1980), with a trace
# ----------------------------------------------------------------------------
def _is_cons(w, i):
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _is_cons(w, i - 1)
    return True


def _measure(s):
    """m in Porter's [C](VC)^m[V] form: number of VC sequences."""
    form = "".join("c" if _is_cons(s, i) else "v" for i in range(len(s)))
    return len(re.findall(r"v+c+", form))


def _has_vowel(s):
    return any(not _is_cons(s, i) for i in range(len(s)))


def _double_cons(w):
    return len(w) >= 2 and w[-1] == w[-2] and _is_cons(w, len(w) - 1)


def _cvc(w):
    return (
        len(w) >= 3
        and _is_cons(w, len(w) - 3)
        and not _is_cons(w, len(w) - 2)
        and _is_cons(w, len(w) - 1)
        and w[-1] not in "wxy"
    )


def _replace(w, rules, min_m):
    """Apply the longest matching suffix rule if m(stem) > min_m."""
    for suf, rep in sorted(rules, key=lambda r: -len(r[0])):
        if w.endswith(suf):
            stem = w[: -len(suf)]
            return stem + rep if _measure(stem) > min_m else w
    return w


STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
         ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
         ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
         ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
         ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")]
STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
         ("ical", "ic"), ("ful", ""), ("ness", "")]
STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
         "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]


def step1a(w):
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def step1b(w):
    if w.endswith("eed"):
        return w[:-1] if _measure(w[:-3]) > 0 else w
    for suf in ("ed", "ing"):
        if w.endswith(suf) and _has_vowel(w[: -len(suf)]):
            w = w[: -len(suf)]
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if _double_cons(w) and w[-1] not in "lsz":
                return w[:-1]
            if _measure(w) == 1 and _cvc(w):
                return w + "e"
            return w
    return w


def step1c(w):
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


def step4(w):
    for suf in sorted(STEP4, key=len, reverse=True):
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if suf == "ion" and not stem.endswith(("s", "t")):
                return w
            return stem if _measure(stem) > 1 else w
    return w


def step5a(w):
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            return stem
    return w


def step5b(w):
    if _measure(w) > 1 and _double_cons(w) and w.endswith("l"):
        return w[:-1]
    return w


STEPS = [
    ("1a", step1a),
    ("1b", step1b),
    ("1c", step1c),
    ("2", lambda w: _replace(w, STEP2, 0)),
    ("3", lambda w: _replace(w, STEP3, 0)),
    ("4", step4),
    ("5a", step5a),
    ("5b", step5b),
]


def my_porter(word, trace=None):
    w = word.lower()
    if len(w) <= 2:
        return w
    for name, fn in STEPS:
        new = fn(w)
        if trace is not None and new != w:
            trace.append((name, w, new))
        w = new
    return w


# ----------------------------------------------------------------------------
# Demo 1: same words, five stemmers
# ----------------------------------------------------------------------------
WORDS_1 = [
    "generalization", "relational", "conditional", "university", "universe",
    "transparent", "sportingly", "happily", "studies", "agreed",
    "caresses", "ponies", "motoring", "hopeful", "probably",
    "organization", "argument", "mice", "better", "flying",
]


def demo1():
    banner("DEMO 1: the same words through five stemmers")
    cols = list(STEMMERS)
    print(f"{'word':<16}" + "".join(f"{c:<16}" for c in cols))
    print("-" * (16 + 16 * len(cols)))
    for w in WORDS_1:
        print(f"{w:<16}" + "".join(f"{STEMMERS[c](w):<16}" for c in cols))
    diff_orig_snow = sum(porter_orig.stem(w) != snowball.stem(w) for w in WORDS_1)
    diff_lan = sum(porter_orig.stem(w) != lancaster.stem(w) for w in WORDS_1)
    print(f"\nPorter(orig) vs Snowball : differ on {diff_orig_snow}/{len(WORDS_1)} words")
    print(f"Porter(orig) vs Lancaster: differ on {diff_lan}/{len(WORDS_1)} words")
    print("avg stem length:  " + "   ".join(
        f"{c}={sum(len(STEMMERS[c](w)) for w in WORDS_1) / len(WORDS_1):.1f}" for c in cols))


# ----------------------------------------------------------------------------
# Demo 2: where stemming gets it wrong
# ----------------------------------------------------------------------------
SHOULD_MERGE = [
    ("data", "datum"), ("matrix", "matrices"), ("analysis", "analyses"),
    ("run", "ran"), ("believe", "belief"), ("mouse", "mice"),
    ("good", "better"), ("sing", "sang"),
]
SHOULD_NOT_MERGE = [
    ("university", "universe"), ("organ", "organization"), ("news", "new"),
    ("police", "policy"), ("wand", "wander"), ("arm", "army"),
    ("past", "paste"), ("heal", "health"),
]


def demo2():
    banner("DEMO 2: where stemmers make mistakes (Porter original vs Snowball vs Lancaster)")
    fns = {"Porter": porter_orig.stem, "Snowball": snowball.stem, "Lancaster": lancaster.stem}

    print("\nUNDER-stemming: related words that SHOULD share a stem")
    print(f"{'pair':<24}" + "".join(f"{n:<26}" for n in fns))
    for a, b in SHOULD_MERGE:
        row = f"{a + ' / ' + b:<24}"
        for fn in fns.values():
            sa, sb = fn(a), fn(b)
            row += f"{sa + ',' + sb:<18}{'OK' if sa == sb else 'MISS':<8}"
        print(row)

    print("\nOVER-stemming: unrelated words that should NOT share a stem")
    print(f"{'pair':<24}" + "".join(f"{n:<26}" for n in fns))
    over = {n: 0 for n in fns}
    for a, b in SHOULD_NOT_MERGE:
        row = f"{a + ' / ' + b:<24}"
        for n, fn in fns.items():
            sa, sb = fn(a), fn(b)
            bad = sa == sb
            over[n] += bad
            row += f"{sa + ',' + sb:<18}{'WRONG' if bad else 'ok':<8}"
        print(row)
    print("\nover-stemming errors: " + ", ".join(f"{n}={c}/{len(SHOULD_NOT_MERGE)}" for n, c in over.items()))


# ----------------------------------------------------------------------------
# Demo 3: inside Porter: my own implementation vs NLTK
# ----------------------------------------------------------------------------
TRACE_WORDS = ["generalizations", "relational", "hopefulness", "agreed", "controlling", "happily"]

BIG_LIST = (
    "caresses ponies ties cats feed agreed plastered bled motoring sing conflated "
    "troubled sized hopping tanned falling hissing fizzed failing filing happy sky "
    "relational conditional rational valenci hesitanci digitizer conformabli "
    "radicalli differentli vileli analogousli vietnamization predication operator "
    "feudalism decisiveness hopefulness callousness formaliti sensitiviti sensibiliti "
    "triplicate formative formalize electriciti electrical hopeful goodness revival "
    "allowance inference airliner gyroscopic adjustable defensible irritant "
    "replacement adjustment dependent adoption homologou communism activate "
    "angulariti homologous effective bowdlerize probate rate cease controll roll "
    "connection connected connecting connections generalization generalizations "
    "organization organizational university universities computation computational "
    "retrieval retrieves relevance relevant documents indexing indexed searches "
    "queries ranking stemming stemmer stemmers algorithm algorithms linguistics "
    "information database databases network networks library libraries studies "
    "studying studied running runner runs happily happiness unhappy argument arguments "
    "arguing nationalization internationalization responsibilities beautifully"
).split()


def demo3():
    banner("DEMO 3: inside Porter: my own implementation, step by step")
    for w in TRACE_WORDS:
        trace = []
        final = my_porter(w, trace)
        print(f"\n{w}")
        for step, before, after in trace:
            print(f"   step {step:<3}: {before:<18} -> {after}")
        print(f"   FINAL     : {final}")

    print("\n" + "-" * 78)
    # print("Check: my implementation vs NLTK's ORIGINAL_ALGORITHM on a word list")
    ok, bad = 0, []
    for w in BIG_LIST:
        mine, ref = my_porter(w), porter_orig.stem(w)
        if mine == ref:
            ok += 1
        else:
            bad.append((w, mine, ref))
    # print(f"words tested: {len(BIG_LIST)}   identical: {ok}   "
    #       f"agreement: {100 * ok / len(BIG_LIST):.1f}%")
    for w, mine, ref in bad[:10]:
        print(f"   mismatch  {w:<20} mine={mine:<14} nltk={ref}")

    print("\nMeasure m of some words (the key condition in Porter's rules):")
    for w in ["tr", "ee", "tree", "by", "trouble", "oats", "trees", "ivy", "troubles", "private", "oaten", "orrery"]:
        print(f"   m({w:<9}) = {_measure(w)}")


# ----------------------------------------------------------------------------
# Demo 4: speed
# ----------------------------------------------------------------------------
def make_text(n_tokens, seed=1):
    rng = random.Random(seed)
    vocab = sorted(set(BIG_LIST + WORDS_1))
    return [rng.choice(vocab) for _ in range(n_tokens)]


def demo4(n=60_000):
    banner(f"DEMO 4: speed on {n:,} tokens")
    tokens = make_text(n)
    results = []
    runners = dict(STEMMERS)
    runners["My Porter (pure py)"] = my_porter
    for name, fn in runners.items():
        t0 = time.perf_counter()
        for tok in tokens:
            fn(tok)
        results.append((name, time.perf_counter() - t0))
    if PyStemmer:
        ps = PyStemmer.Stemmer("english")
        t0 = time.perf_counter()
        ps.stemWords(tokens)
        results.append(("PyStemmer (C Snowball)", time.perf_counter() - t0))
    else:
        print("(PyStemmer not installed: skipping the C version)")

    fastest = min(t for _, t in results)
    print(f"\n{'stemmer':<24}{'time (s)':>10}{'words/sec':>14}{'vs fastest':>12}")
    for name, t in sorted(results, key=lambda r: r[1]):
        print(f"{name:<24}{t:>10.3f}{n / t:>14,.0f}{t / fastest:>11.1f}x")


# ----------------------------------------------------------------------------
# Demo 5: why IR cares: vocabulary size + search
# ----------------------------------------------------------------------------
DOCS = {
    "d1": "The runner was running through the park while the dogs were barking",
    "d2": "She runs every morning and studies at the library",
    "d3": "Connected computers share data through network connections",
    "d4": "The universities organized a conference on computational linguistics",
    "d5": "He studied the generalization of relational databases",
    "d6": "Running shoes are designed for comfortable runs",
    "d7": "Computing connections between documents is the goal of information retrieval",
    "d8": "The retrieval system retrieves relevant documents for each query",
    "d9": "The new university published the news about universal healthcare",
    "d10": "A police officer explained the new policy in the universe of law",
}
STOP = set("the a an of and is are was were to in on at for by with through while "
           "every each she he it between about as".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOP]


def build_index(stem_fn):
    idx = defaultdict(set)
    for doc_id, text in DOCS.items():
        for tok in tokenize(text):
            idx[stem_fn(tok)].add(doc_id)
    return idx


def search(idx, stem_fn, query):
    hits = set()
    for tok in tokenize(query):
        hits |= idx.get(stem_fn(tok), set())
    return sorted(hits, key=lambda d: int(d[1:]))


def demo5():
    banner("DEMO 5: why IR cares: vocabulary size and search results")
    methods = {"No stemming": lambda w: w, **{k: v for k, v in STEMMERS.items() if k != "Porter(Martin)"}}
    indexes = {name: build_index(fn) for name, fn in methods.items()}

    base = len(indexes["No stemming"])
    print(f"\n{len(DOCS)} documents. Index vocabulary size per method:")
    for name, idx in indexes.items():
        print(f"   {name:<14}{len(idx):>4} terms   ({100 * (1 - len(idx) / base):4.1f}% smaller)")

    queries = [
        ("run", "d1 d2 d6"),
        ("connection", "d3 d7"),
        ("retrieved", "d7 d8"),
        ("study", "d2 d5"),
        ("university", "d4 d9"),
        ("news", "d9"),
    ]
    print("\nQuery results (ideal answer in brackets):")
    for q, ideal in queries:
        print(f"\n   query: '{q}'   [ideal: {ideal}]")
        ideal_set = set(ideal.split())
        for name, fn in methods.items():
            hits = search(indexes[name], fn, q)
            extra = [h for h in hits if h not in ideal_set]
            missing = [h for h in ideal_set if h not in hits]
            note = ""
            if extra:
                note += f"  false hits: {' '.join(extra)}"
            if missing:
                note += f"  missed: {' '.join(sorted(missing))}"
            print(f"      {name:<14}-> {' '.join(hits) or '(nothing)':<20}{note}")


# ----------------------------------------------------------------------------
# Demo 6 (bonus): stemming vs lemmatization
# ----------------------------------------------------------------------------
def demo6():
    # banner("DEMO 6 (bonus): stemming vs lemmatization")
    # try:
    #     from nltk.stem import WordNetLemmatizer
    #     import nltk
    #     lem = WordNetLemmatizer()
    #     try:
    #         lem.lemmatize("test")
    #     except LookupError:
    #         nltk.download("wordnet", quiet=True)
    #         nltk.download("omw-1.4", quiet=True)
    #         lem.lemmatize("test")
    # except Exception as e:  # no internet / no wordnet
    #     print(f"WordNet not available ({type(e).__name__}); skipping.")
    #     print("Fix once with:  python -c \"import nltk; nltk.download('wordnet')\"")
    #     return
    # items = [("better", "a"), ("mice", "n"), ("was", "v"), ("feet", "n"),
    #          ("studies", "v"), ("running", "v"), ("went", "v"), ("geese", "n")]
    # print(f"\n{'word':<10}{'Porter':<12}{'Snowball':<12}{'Lancaster':<12}{'Lemma':<10}")
    # for w, pos in items:
    #     print(f"{w:<10}{porter_orig.stem(w):<12}{snowball.stem(w):<12}"
    #           f"{lancaster.stem(w):<12}{lem.lemmatize(w, pos=pos):<10}")
    # print("\nStemmers cut suffixes by rule (fast, may give non-words); "
    #       "lemmatizers use a dictionary (slower, real words).")
    return


DEMOS = {1: demo1, 2: demo2, 3: demo3, 4: demo4, 5: demo5, 6: demo6}


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--demo", type=int, choices=DEMOS.keys())
    args = ap.parse_args()
    PAUSE = not args.no_pause

    todo = [args.demo] if args.demo else list(DEMOS)
    for i, n in enumerate(todo):
        DEMOS[n]()
        if i < len(todo) - 1:
            wait()
    print("\nDone.")


if __name__ == "__main__":
    main()