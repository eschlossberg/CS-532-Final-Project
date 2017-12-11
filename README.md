# CS-532-Final-Project

## Milestone checker
- [ ] !important Test & Debug Doc2Vec implementation
- [ ] Information on Doc2Vec's mathematical magic
- [x] Implementation Doc2Vec's algorithm for this project
- [x] Pegasos implementation

## [The Magic behind Doc2Vec](https://www.quora.com/How-does-doc2vec-represent-feature-vector-of-a-document-Can-anyone-explain-mathematically-how-the-process-is-done/answer/Piyush-Bhardwaj-7)

The goal of doc2vec is to create a vector representation of a document. Doc2vec utilizes word2vec model (where word is representated as a feature vector in this model) and added a small extension - another document vector. The Doc2Vec algorithm will train both word vectors and document vectors at the same time (the vectors are averaged or concatenated) to form a vector representation of the document. This document vector intends to represent the concept of this document.

[Link to this document](https://towardsdatascience.com/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)

1.	An unsupervised training algorithm – “Paragraph Vectors” algorithm – to create a numeric representation of a document
2.	Intention: encode whole document that consist lists of grouped sentences and associate with label

> Doc2Vec’s learning strategy exploits the idea that the prediction of neighboring words for a given word strongly relies on the document also. Though the appearance of the phrase catch the ball is frequent in the corpus, if we know that the topic of a document is about ‘’technology’’, we can expect words such as bug or exception after the word catch (ignoring the) instead of the word ball since catch the bug/exception is more plausible under the topic ‘’technology’’. On the other hand, if the topic of the document is about ‘’sports’’, then we can expect ball after catch.

3.	Better efficiency than “Bags of words” approach
4.	[“Paragraph Vector”](https://arxiv.org/pdf/1405.4053.pdf) –
    -	Learned from unlabeled data – solves insufficient labeled data
    - Inherit semantics of the words
	  - The word vectors are averaged or concatenated to form document vector - to predict the next word in a context.
    - cosine-similarity between examples are likely to be a useful measure of similarity.
