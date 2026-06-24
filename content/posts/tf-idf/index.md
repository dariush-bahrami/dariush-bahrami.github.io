+++
date = '2026-06-24T12:00:14+03:30'
draft = false
title = 'TF-IDF Explained'
description = 'TF-IDF From Scratch: Teaching a Computer to Find the Right Document'
tags = ['natural-language-processing', 'information-retrieval', 'python', 'from-scratch']
math = true
+++

Search feels like magic until you build one yourself. Type in a few words, and somehow the right document floats to the top of the list. Under the hood, even the simplest version of this magic relies on an elegant, decades-old idea: TF-IDF. In this post, we'll build a tiny search engine from scratch, using nothing but Python's standard library, and use it to find the most relevant document for a query.

## The Setup

Imagine we have ten short news snippets sitting in text files on disk — a mix of stories about machine learning, space exploration, and health:

```
doc_01.txt: Machine learning algorithms are transforming the tech industry with neural networks.
doc_02.txt: Deep learning and artificial intelligence are driving modern automation systems.
doc_03.txt: The tech industry relies heavily on AI to process massive datasets.
doc_04.txt: NASA launches a new rover to explore the surface of Mars.
doc_05.txt: Astronomers discover a new exoplanet in the habitable zone of a distant star.
doc_06.txt: The space agency telescope captured stunning new images of a distant galaxy.
doc_07.txt: Doctors recommend a balanced diet and regular exercise to maintain heart health.
doc_08.txt: A new clinical trial for an asthma treatment shows promising results for patients.
doc_09.txt: Cardiovascular health is strongly linked to daily exercise and lower stress levels.
doc_10.txt: A new artificial heart valve was successfully implanted in the patient using automated robotic surgery.
```

Our goal: given a query like `"automated heart surgery"`, figure out which document is the best match — without any machine learning model, just arithmetic.

## Comparing Text With Cosine Similarity

To rank documents against a query, we need a notion of *similarity*. A natural choice is cosine similarity, which measures the angle between two vectors rather than their raw distance:

$$\text{Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

The numerator is the dot product of the two vectors; the denominator normalizes by their magnitudes. The result is a number between -1 and 1 that captures how aligned two vectors are, regardless of their length.

This is great — except our query and documents are sentences, not vectors. Before we can compare anything, we need a way to turn text into numbers.

## From Words to Vectors: The TF-IDF Idea

There are many ways to convert text into vectors, ranging from simple counting methods to deep learning models like BERT. TF-IDF is one of the classic approaches, and it rests on a few simplifying assumptions:

1. Text is made of discrete units called tokens (roughly, words).
2. There's a fixed, predefined vocabulary — the set of all tokens we care about.
3. Word order doesn't matter.
4. Not all words are equally informative.

That last point is the heart of the whole technique, so let's unpack it.

### Term Frequency: how much of this word is here?

Suppose our vocabulary is just four words: `["apple", "cat", "dog", "orange"]`. For any document, we can build a 4-dimensional vector where each slot answers one question: *what fraction of this document is made up of this word?*

$$TF = \frac{\text{Count of the token in the document}}{\text{Total words in the document}}$$

This is called the **term frequency** vector. It's simple, and it's already usable inside the cosine similarity formula. But it has a blind spot.

Consider the query `"a cat chased the mouse"`. If we score documents using term frequency alone, the dot product treats every word the same way:

$$\mathbf{A} \cdot \mathbf{B} = a_1 b_1 + a_2 b_2 + a_3 b_3 + a_4 b_4$$

The word `"a"` contributes to the score exactly as much as the word `"cat"`. That's a problem, because intuitively, `"cat"` tells us something specific about the document, while `"a"` tells us almost nothing. But it's worth pausing on *why*, exactly, that's true.

Here's one way to frame it: the word `"cat"` is specific. If someone tells you "this text contains the word 'cat'," you instantly have a bunch of assumptions and expectations about that text — it's probably about animals, pets, or maybe biology. Compare that to being told "this text contains the word 'a'." You've learned almost nothing, because that's true of nearly every piece of English text ever written. The amount of information you gain from being told a word is present depends entirely on how surprising that fact is — and a word is only surprising if it's rare.

That's the whole idea, compressed into one word: **rarity**.

### Inverse Document Frequency: how rare is this word?

A word that appears in nearly every document carries little information, by the logic above, while a word that appears in only a few documents is far more distinctive. We can capture this with the inverse document frequency:

$$IDF = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing the word}}\right)$$

The total document count is fixed, so the whole expression depends on how many documents contain the word. If a word appears in almost every document, the fraction inside the log is close to 1, and $\log(1) = 0$ — the word gets almost no weight. If a word is rare, the fraction is large, the log is a sizable positive number, and the word gets a strong weight.

Crucially, IDF is computed once per vocabulary, using the whole document collection — it doesn't change document by document. It's a fixed "importance score" for every word we know about.

### Putting it together: TF-IDF

Multiply each word's term frequency by its inverse document frequency, and you get the TF-IDF vector:

$$TF\text{-}IDF = TF \times IDF$$

This single vector now balances two forces: how *often* a word shows up in a specific document, and how *rare* that word is across all documents. Common filler words get suppressed; distinctive, topic-specific words get amplified.

## Building It, Step by Step

With the theory in place, here's the recipe:

**For the document collection (computed once):**
1. Tokenize every document into words.
2. Build the vocabulary — the set of all unique tokens across all documents.
3. Compute the IDF value for every token in the vocabulary.
4. Compute the TF vector for each document.
5. Multiply each document's TF vector by the shared IDF vector to get its TF-IDF vector.

**For the query:**
1. Tokenize it.
2. Compute its TF vector against the same vocabulary.
3. Multiply by the cached IDF vector to get the query's TF-IDF vector.
4. Compute cosine similarity between the query vector and every document vector.
5. Sort by similarity, and the top result is your answer.

Let's implement each piece.

### Tokenization

Tokenization just means splitting text into a clean list of words. A minimal version: lowercase everything, strip punctuation, then split on whitespace.

```python
from string import punctuation

def tokenize(document: str) -> list[str]:
    document = document.lower()
    for character in punctuation:
        document = document.replace(character, " ")
    return document.split()
```

Running this on `doc_01.txt` turns *"Machine learning algorithms are transforming the tech industry with neural networks."* into a clean list:

```python
['machine', 'learning', 'algorithms', 'are', 'transforming',
 'the', 'tech', 'industry', 'with', 'neural', 'networks']
```

No punctuation, no capitalization quirks — just words.

### Building the Vocabulary

Once every document is tokenized, the vocabulary is simply the set of every unique word seen anywhere in the collection:

```python
vocabulary = set()
for document_tokens in filename_to_tokenized_documents.values():
    vocabulary.update(document_tokens)
vocabulary = sorted(vocabulary)
```

Across our ten snippets, this produces 87 unique tokens — everything from `"a"` and `"agency"` to far more specific words like `"asthma"` and `"astronomers"`.

### Computing IDF

For each word in the vocabulary, we count how many documents contain it, then apply the IDF formula:

```python
from math import log

def get_inverse_document_frequency_vector(
    vocabulary: list[str], tokenized_documents: list[list[str]]
) -> list[float]:
    total_documents = len(tokenized_documents)
    inverse_document_frequency_vector = []
    for token in vocabulary:
        number_of_documents_with_token = sum(
            1 for document_tokens in tokenized_documents if token in document_tokens
        )
        inverse_document_frequency_vector.append(
            log(total_documents / number_of_documents_with_token)
        )
    return inverse_document_frequency_vector
```

This gives us one IDF value per vocabulary word, computed once and reused for every document and every future query.

### Computing TF

Term frequency is computed per document: for each word in the vocabulary, count its occurrences in that document and divide by the document's total word count.

```python
def get_term_frequency_vector(
    document_tokens: list[str], vocabulary: list[str]
) -> list[float]:
    total_tokens_in_document = len(document_tokens)
    term_frequency_vector = []
    for token in vocabulary:
        token_count_in_document = document_tokens.count(token)
        term_frequency_vector.append(token_count_in_document / total_tokens_in_document)
    return term_frequency_vector
```

### Combining TF and IDF

This part is almost anticlimactic after all the setup — just multiply corresponding elements:

```python
def get_tf_idf_vector(
    term_frequency_vector: list[float], inverse_document_frequency_vector: list[float]
) -> list[float]:
    return [tf * idf for tf, idf in zip(term_frequency_vector, inverse_document_frequency_vector)]
```

We run this for every document, and also for the query itself — using the same cached IDF vector — so everything lives in the same vector space and can be fairly compared.

### Cosine Similarity

Finally, the function that ties it all together:

```python
from math import sqrt

def get_cosine_similarity(vector_1: list[float], vector_2: list[float]) -> float:
    vector_1_magnitude = sqrt(sum(x**2 for x in vector_1))
    vector_2_magnitude = sqrt(sum(x**2 for x in vector_2))
    dot_product = sum(x * y for x, y in zip(vector_1, vector_2))
    return dot_product / (vector_1_magnitude * vector_2_magnitude)
```

## Putting It to the Test

With every piece in place, we run the full pipeline against the query `"automated heart surgery"` and score it against all ten documents:

| Document | Relevance Score |
|---|---|
| doc_01 – doc_06 | 0.0 |
| doc_07 (heart health, exercise) | 0.110 |
| doc_08 | 0.0 |
| doc_09 | 0.0 |
| **doc_10 (artificial heart valve, automated surgery)** | **0.483** |

The winner, by a wide margin, is `doc_10.txt`:

> *"A new artificial heart valve was successfully implanted in the patient using automated robotic surgery."*

This makes intuitive sense — it shares both `"automated"` and the conceptual core of `"heart surgery"` with the query, and those words are rare enough across the collection to carry real weight. `doc_07`, which mentions heart health and exercise, gets a small but nonzero score thanks to the overlapping word `"heart"`. Every document about machine learning, space, or unrelated health topics scores exactly zero — they share no vocabulary with the query at all.

## Why This Still Matters

It's tempting to think TF-IDF is a relic, made obsolete by transformer-based embeddings that capture meaning rather than just word overlap. And it's true that TF-IDF won't realize `"automated heart surgery"` is conceptually close to `"robotic cardiac procedure"` — it only sees the literal words.

But TF-IDF remains genuinely useful: it's fast, requires no training data or GPU, is fully interpretable (you can see exactly *why* a document scored the way it did), and works surprisingly well for keyword-style search, deduplication, and as a baseline to beat when evaluating fancier models. Many production search and recommendation systems still use it, often combined with more modern techniques.

More importantly, building it from scratch demystifies something that can otherwise feel like a black box. Once you've written the IDF formula yourself and watched it suppress the word `"a"` while boosting the word `"heart"`, you understand *why* search engines behave the way they do — not just that they do.

The next natural step from here is to ask: what happens when the query and the most relevant document don't share any exact words at all? That's exactly the kind of gap that word embeddings and neural retrieval models were built to close — but that's a story for another post.
