"""Character-level byte-pair encoding, trained from scratch in plain Python.

Usage: python3 -I train.py path/to/input.txt [n_merges]

Writes, next to this file:
  train.log        one line per merge: step, pair, new token, pair count, corpus tokens after
  model.json       compact export (base characters, merges, counts, curve, stats)
"""
import json, os, re, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# GPT-2-style pre-split, restricted to explicit ASCII classes so that the
# Python and JavaScript regex engines agree exactly. Merges never cross chunks.
#   optional space + letters | optional space + digits | optional space + other
#   | whitespace run that is not followed by a non-space (keeps the last space
#     for the next word) | any remaining whitespace run
WS = " \\t\\n\\r"
PATTERN = (f" ?[A-Za-z]+| ?[0-9]+| ?[^{WS}A-Za-z0-9]+"
           f"|[{WS}]+(?![^{WS}])|[{WS}]+")
SPLIT = re.compile(PATTERN)


def pretokenize(text):
    return SPLIT.findall(text)


def train(text, n_merges):
    t0 = time.time()
    base = sorted(set(text))                      # base vocabulary: every character seen
    cid = {c: i for i, c in enumerate(base)}
    chunk_freq = Counter(pretokenize(text))
    words = [[cid[c] for c in w] for w in chunk_freq]
    freqs = list(chunk_freq.values())
    vocab = list(base)                            # id -> string

    pair_count = defaultdict(int)
    where = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for a, b in zip(w, w[1:]):
            pair_count[(a, b)] += f
            where[(a, b)].add(wi)

    total = sum(len(w) * f for w, f in zip(words, freqs))
    assert total == len(text)
    merges, counts, after = [], [], []
    for step in range(n_merges):
        if not pair_count:
            break
        # highest count wins; ties go to the smallest (a, b) ids, deterministic
        (a, b), cnt = max(pair_count.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if cnt < 2:
            break
        new = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        for wi in list(where[(a, b)]):
            w = words[wi]
            f = freqs[wi]
            # remove this word's old pair contributions
            for p in zip(w, w[1:]):
                pair_count[p] -= f
                if pair_count[p] == 0:
                    del pair_count[p]
                where[p].discard(wi)
            # merge left to right, non-overlapping
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new); i += 2
                else:
                    out.append(w[i]); i += 1
            total -= (len(w) - len(out)) * f
            words[wi] = out
            for p in zip(out, out[1:]):
                pair_count[p] += f
                where[p].add(wi)
        where.pop((a, b), None)
        merges.append((a, b)); counts.append(cnt); after.append(total)
    return dict(base=base, merges=merges, counts=counts, after=after, vocab=vocab,
                chunks=len(chunk_freq), seconds=time.time() - t0)


def encode(text, base, merges, k=None):
    """Tokenize with the first k merges. Unknown characters get id -1."""
    if k is None:
        k = len(merges)
    cid = {c: i for i, c in enumerate(base)}
    rank = {tuple(m): r for r, m in enumerate(merges[:k])}
    nb = len(base)
    ids, strs = [], []
    for chunk in pretokenize(text):
        sym = [cid.get(c, -1) for c in chunk]
        s = list(chunk)
        while len(sym) > 1:
            best, br = None, None
            for p in zip(sym, sym[1:]):
                r = rank.get(p)
                if r is not None and (br is None or r < br):
                    best, br = p, r
            if best is None:
                break
            new = nb + br
            o_sym, o_s, i = [], [], 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == best[0] and sym[i + 1] == best[1]:
                    o_sym.append(new); o_s.append(s[i] + s[i + 1]); i += 2
                else:
                    o_sym.append(sym[i]); o_s.append(s[i]); i += 1
            sym, s = o_sym, o_s
        ids += sym; strs += s
    return ids, strs


def show(s):
    return s.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥").replace("\r", "␍")


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    text = open(path, encoding="utf-8").read()
    r = train(text, n)
    base, merges, counts, after, vocab = r["base"], r["merges"], r["counts"], r["after"], r["vocab"]
    nb = len(base)

    # sanity: re-encoding the whole training text with the final merges gives the same token count
    ids, _ = encode(text, base, merges)
    assert len(ids) == after[-1], (len(ids), after[-1])
    assert all(i >= 0 for i in ids)

    with open(os.path.join(HERE, "train.log"), "w", encoding="utf-8") as f:
        f.write(f"# char-level BPE on {os.path.basename(path)}: {len(text)} chars, {nb} base characters, "
                f"{r['chunks']} distinct pre-split chunks, {len(merges)} merges, {r['seconds']:.1f}s\n")
        f.write(f"# pre-split regex: {PATTERN}\n")
        f.write("# step\tleft\tright\tnew_token\tnew_id\tpair_count\tcorpus_tokens_after\n")
        for i, ((a, b), c, t) in enumerate(zip(merges, counts, after)):
            f.write(f"{i+1}\t{show(vocab[a])}\t{show(vocab[b])}\t{show(vocab[nb+i])}\t{nb+i}\t{c}\t{t}\n")

    lens = [len(v) for v in vocab[nb:]]
    longest = max(range(len(lens)), key=lambda i: (lens[i], -i))
    word_tokens = sum(1 for v in vocab[nb:] if re.fullmatch(r" ?[A-Za-z]+", v))
    model = {
        "kind": "char-level BPE",
        "corpus": "tiny-shakespeare (input.txt from karpathy/char-rnn)",
        "pattern": PATTERN,
        "base": "".join(base),
        "merges": [x for m in merges for x in m],   # flat [a0,b0,a1,b1,...]
        "counts": counts,
        "after": after,
        "stats": {
            "chars": len(text),
            "base": nb,
            "chunks": r["chunks"],
            "merges": len(merges),
            "vocab": len(vocab),
            "tokens_final": after[-1],
            "chars_per_token_final": round(len(text) / after[-1], 3),
            "first_count": counts[0],
            "last_count": counts[-1],
            "longest_token": vocab[nb + longest],
            "longest_step": longest + 1,
            "word_like_tokens": word_tokens,
            "seconds": round(r["seconds"], 1),
        },
    }
    with open(os.path.join(HERE, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, separators=(",", ":"))
    print(json.dumps(model["stats"], indent=1))
    for i in range(12):
        a, b = merges[i]
        print(i + 1, repr(vocab[a]), "+", repr(vocab[b]), "->", repr(vocab[nb + i]), counts[i])


if __name__ == "__main__":
    main()
