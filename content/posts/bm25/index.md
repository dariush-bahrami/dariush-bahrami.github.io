+++
date = '2026-07-03T16:58:36+03:30'
draft = false
title = 'BM25 Explained'
description = "BM25 Explained: Fixing TF-IDF's Two Biggest Blind Spots"
tags = ['natural-language-processing', 'information-retrieval', 'search']
math = true
+++

I already wrote a [blog post about TF-IDF](https://dariush-bahrami.github.io/posts/tf-idf/). I highly recommend reading it before this one.

The goal of TF-IDF is to give us a **matching score** between two documents. Using this score, given a query and a bunch of documents, we can compute how well the query matches each document in our database and return the most relevant results.

Here's the gist of what TF-IDF does:

> It asks: how much do these two documents overlap on *informative* keywords?

So the score comes from two factors, computed per keyword:

1. **How much does a document use a given keyword?** — this is the **TF** (Term Frequency) part.
2. **How informative is that keyword?** — this is the **IDF** (Inverse Document Frequency) part.

### Term Frequency (TF)

TF is straightforward. For a given keyword $t$ and document $d$, we compute what fraction of the document is made up of that keyword:

$$
\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total number of words in } d}
$$

We compute this for every keyword $t$ in our vocabulary, for both the document and the query. The intuition: a document that uses a given word frequently and substantially is more likely to consider that word important — and if the query also weighs that word heavily, that's a sign of a strong match.

### Inverse Document Frequency (IDF)

IDF scales each keyword's contribution to the final score. It gives **more weight to rare keywords**, because rare words carry more information, and **less weight to common keywords**, because they carry less.

To see why rare words are more informative, try this mind game: imagine I have a text I haven't shown you yet, and I let you guess what it's about by revealing a few of its words, one at a time.

If I tell you the text contains the words **"cat"** and **"farm"**, you immediately form a pretty good guess about what the text is about.

Now imagine I'd instead told you the text contains the words **"the"** and **"is"**. That tells you almost nothing — nearly every document in existence contains those words.

This is exactly the role IDF plays: it rewards rarity, because the more *surprising* a word is, the more information it gives us. Formally, for a keyword $t$ across a collection of $N$ documents, where $n_t$ is the number of documents containing $t$:

$$
\text{IDF}(t) = \log\left(\frac{N}{n_t}\right)
$$

When $t$ shows up in nearly every document ($n_t \approx N$), the ratio $N/n_t$ is close to $1$, and $\log(1) = 0$ — so the word contributes almost nothing to the score, just like "the" or "is" in our mind game. When $t$ is rare ($n_t \ll N$), the ratio is large, the log is large, and the word contributes a lot — just like "cat" or "farm."

### From scores to vectors

Putting TF and IDF together for a single term gives us a per-term weight:

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)
$$

Now do this for *every* term $t$ in our vocabulary, not just the ones a document happens to contain (terms the document doesn't have simply get a weight of $0$). Stacking these per-term weights together turns each document into a vector:

$$
\vec{v}_d = \big(\text{TF-IDF}(t_1, d),\ \text{TF-IDF}(t_2, d),\ \dots,\ \text{TF-IDF}(t_n, d)\big)
$$

We treat the query $q$ the same way, as if it were a tiny document, giving us a query vector $\vec{v}_q$.

### The matching score: cosine similarity

To get a single matching score between the query and a document, we compare their two vectors using **cosine similarity**:

$$
\text{score}(q, d) = \cos(\vec{v}_q, \vec{v}_d) = \frac{\vec{v}_q \cdot \vec{v}_d}{\lVert \vec{v}_q \rVert \, \lVert \vec{v}_d \rVert}
$$

The dot product on top naturally only picks up contributions from terms that appear in *both* vectors — any term missing from one side contributes a $0$ to that term's product. The division by the norms $\lVert \vec{v}_q \rVert \lVert \vec{v}_d \rVert$ is what makes this *cosine* similarity rather than a plain dot product: it normalizes away document length, so a long document doesn't win the matching contest just by containing more words.

## TF-IDF Problems

TF-IDF is an elegant, simple algorithm for measuring how well two documents match. But it has its own limitations that need fixing.

One major flaw is the **keyword-spamming problem**. Consider this situation: our query contains the keyword "tree." We examine our documents and find that just three out of 100 documents contain the word "tree" at all. One of them mentions "tree" just once. The second mentions it 5 times. The third mentions it 100 times.

Now ask yourself: which of these three documents is *most relevant* to the query "tree"?

If we hand this question to TF-IDF, it answers with a straight line: relevance scales **linearly** with mention count. Assuming, for the sake of this example, that all three documents are roughly the same length, this means the document with 100 mentions scores about 20 times higher than the document with 5 mentions.

Do you see the problem? Once a document mentions a keyword a handful of times, we already know it's *about* that keyword. Mentioning it 100 times doesn't make a document 20 times more relevant — it could just as easily be keyword spam.

The second problem is that TF-IDF doesn't care how *long* a document is. A 1,000-page book will naturally mention all sorts of common words simply by virtue of its length, which lets it match almost any query, relevant or not. There's also a subtler version of this same issue: imagine you search "ratatouille recipe," and two documents match with an equal score — one is an entire cookbook, the other is a single focused recipe blog post, and both happen to contain the recipe. Naturally, we'd prefer the focused, to-the-point document. TF-IDF has no way to express that preference.

We can summarize these two problems as follows:

1. **Unbounded Term Frequency** — relevance keeps growing linearly with mention count, with no diminishing returns.
2. **Document Length** — longer documents have an unfair structural advantage, with no normalization for how focused a document actually is.

## BM25

The **Best matching 25** (or BM25 for short) algorithm patches the two TF-IDF problems above directly into the formula. Here it is in full, intimidating as it looks — we'll take it apart term by term right after:

$$
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \dfrac{|D|}{\text{avgdl}}\right)}
$$

Some of this you already know. $Q$ is the query, made of keywords $q_1, \dots, q_n$. The sum over $i$ and the $\text{IDF}(q_i)$ factor are the same idea as TF-IDF: add up a per-keyword contribution, and weight each keyword by how rare it is. $\text{IDF}$ itself gets a new formula, covered below, but its job — reward rare keywords, ignore common ones — hasn't changed.

What's new is everything inside that big fraction, which replaces TF-IDF's plain $\text{TF}(t,d)$. Three new symbols appear there:

- $f(q_i, D)$ — the raw count of keyword $q_i$ in document $D$. TF-IDF's $\text{TF}$ was this count *divided by the document's total word count*; BM25 starts from the undivided count instead and normalizes for length differently, as we'll see.
- $|D|$ — the length of document $D$, i.e. its total word count.
- $\text{avgdl}$ — the average length, in words, of documents in the collection.

With those defined, the fraction breaks into two jobs, matching the two problems from the TF-IDF section.

### Job 1: capping term frequency

TF-IDF scaled linearly with mention count: a document mentioning "tree" 100 times scored 20x higher than one mentioning it 5 times. We want a function that still grows with $f$, but with sharply diminishing returns. Set aside length normalization for a moment ($b=0$, which — as shown below — strips it out) and the fraction reduces to:

$$
\frac{f \cdot (k_1+1)}{f + k_1}
$$

$k_1$ is a constant we pick, typically between $1.2$ and $2.0$; we'll use $k_1=1.5$. Plugging in the mention counts from the "tree" example:

| $f$ | $\dfrac{f \cdot 2.5}{f+1.5}$ |
|---|---|
| 1 | 1.00 |
| 5 | 1.92 |
| 10 | 2.17 |
| 50 | 2.43 |
| 100 | 2.46 |

Going from 1 mention to 5 (a 5x increase) raises the score by 0.92. Going from 10 to 50 (also a 5x increase) raises it by only 0.25 — roughly 3.6x less movement for the same proportional jump. The growth keeps shrinking as $f$ increases, and the whole expression approaches a ceiling of $k_1+1 = 2.5$ as $f \to \infty$, never exceeding it no matter how many times the term appears. That ceiling is exactly why $(k_1+1)$ sits in the numerator: it sets the value this fraction converges to.

$k_1$ controls how fast that ceiling is approached. A smaller $k_1$ reaches it after fewer mentions; a larger $k_1$ stretches the climb out, behaving more like TF-IDF's straight line.

### Job 2: normalizing for document length

TF-IDF ignored length entirely: a word appearing 10 times in a 2,000-word book scored the same as it appearing 10 times in a 50-word post, even though the post is far more clearly *about* that word. BM25 fixes this with the bracketed term:

$$
1 - b + b \cdot \frac{|D|}{\text{avgdl}}
$$

$|D|/\text{avgdl}$ is document $D$'s length relative to the collection's average — $1$ if $D$ is exactly average length, $2$ if twice the average, $0.5$ if half. $b$ is a second constant, between $0$ and $1$ (default $0.75$), that controls how much this ratio is allowed to matter. Check the extremes: at $b=0$, the expression collapses to $1$ regardless of $|D|/\text{avgdl}$ — length is ignored entirely, which is the $b=0$ case used in Job 1 above. At $b=1$, it collapses to exactly $|D|/\text{avgdl}$ — length matters at full strength. For a document twice the average length with $b=0.75$: $1 - 0.75 + 0.75 \times 2 = 1.75$, partway toward full strength.

### Why the two jobs share one denominator

The full denominator is $f + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)$ — Job 1's $k_1$ is multiplied by Job 2's bracket. Note this multiplication happens only in the denominator; the numerator's $(k_1+1)$ stays fixed regardless of document length. That matters: it means every document, long or short, is climbing toward the *same* ceiling of $k_1+1=2.5$ — length doesn't change the ceiling, only how fast a document approaches it.

Concretely, take two documents with the same value of $f$ for some term, one of average length ($|D|/\text{avgdl}=1$, so the bracket is $1$) and one twice the average length ($|D|/\text{avgdl}=2$, $b=0.75$, so the bracket is $1.75$, shown above). At $f=10$ mentions, the average-length document scores $2.174$ — 87% of the way to the $2.5$ ceiling. The longer document, with the same 10 mentions, scores only $1.980$ — 79% of the way there. Same mention count, same ceiling, but the longer document has more ground left to cover, because its larger bracket inflates the denominator. That's the length penalty: not a separate correction bolted on, but the same saturation curve made harder to climb.

### Fixing IDF

TF-IDF's $\log(N/n_t)$ already does the basic job: rare terms score high, common terms score low. BM25 keeps that same goal but uses a different-looking formula, derived from a probabilistic model of relevance rather than picked by intuition:

$$
\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)
$$

We won't re-derive this one fully — the derivation leans on a few assumptions about relevant versus non-relevant documents that are debated even among the people who study this for a living, and walking through them honestly would take more space than it's worth here. What we can do is explain the pieces mechanically.

The core is $\frac{N-n_t+0.5}{n_t+0.5}$: documents *without* the term, divided by documents *with* it — the odds against running into the term at all. A common term, say $n_t=80$ out of $N=100$, gives odds of $\frac{20.5}{80.5}\approx0.255$, low because the term is everywhere. A rare term, $n_t=3$ out of $100$, gives odds of $\frac{97.5}{3.5}\approx27.9$, high because it almost never shows up. Taking the log turns this into a score: small for common terms, large for rare ones — same shape as TF-IDF's IDF.

Two small additions handle edge cases. The $+0.5$ in both the numerator and denominator, already folded into the odds above, prevents division by zero when a term is in no documents, and prevents $\log(0)$ when a term is in every document. The $+1$ inside the log prevents a different problem: once a term appears in more than half the collection, the ratio drops below $1$ and the log goes negative — at $n_t=80$, $\log(0.255)\approx-1.37$. A negative score would mean a document is *punished* for containing a query term, which shouldn't happen; adding $1$ keeps the result at or above zero, since $\ln(0.255+1)=\ln(1.255)\approx0.227$.

### Putting it back together

That's every piece of the formula at the top: $\text{IDF}(q_i)$ rewards rare keywords without ever going negative; $f(q_i,D)$ and $(k_1+1)$ build a score that saturates at a fixed ceiling instead of growing forever; and the $\left(1-b+b\cdot\frac{|D|}{\text{avgdl}}\right)$ term, multiplying $k_1$ in the denominator, makes that ceiling slower to reach for documents longer than average. Same skeleton as TF-IDF — informativeness times frequency, summed over keywords — with both of its weak points patched.

## Wrapping up

BM25 isn't a niche academic idea — it's the default relevance function behind Elasticsearch, Lucene, and Solr, which means a huge share of the search boxes you use every day, from documentation sites to internal company search, are running some version of this formula under the hood. That ubiquity is really a testament to how well the two fixes hold up in practice: capping term frequency stops keyword stuffing from gaming the score, and normalizing for document length keeps a focused, on-topic page competitive with a much longer one that happens to mention the same words in passing.

It's also worth placing BM25 in the bigger picture. Everything in this post and the TF-IDF one before it is a *lexical* method — it matches on exact keyword overlap, so a query for "car" won't match a document that only says "automobile." That's the gap that modern *semantic* or *vector* search methods, built on embeddings from neural networks, are designed to close. In practice, the two approaches are often combined rather than treated as rivals: BM25 catches precise keyword and rare-term matches that embeddings can blur past, while embeddings catch the synonyms and paraphrases that BM25 misses entirely. That combination — usually called **hybrid search** — is a natural next topic, and one I'll probably write about next.