# Word Similarity
Word similarity calculation mainly on the dataset [Mturk-771](http://www2.mta.ac.il/~gideon/mturk771.html).

1. wordnet: path_similarity, wup_similarity, lch_similarity, res_similarity, jcn_similarity, lin_similarity (WordNet-based)
2. SimByGoogleSearch, WebJaccard, WebOverlap, WebDice, WebPMI, NGD (Google Search Based)
3. LSA-Wikipedia, LDA-Wikipedia (Wikipedia-based)
4. Word2Vec, Fasttext, ELMo, BERT (Representation Learning)



| Method          | MTurk-771 (spearmanr) | MEN (spearmanr) | RW-STANFORD(spearmanr) | SimLex-999(spearmanr) | SimVerb-3500(spearmanr) |
| --------------- | --------------------- | --------------- | ---------------------- | --------------------- | ----------------------- |
| $Sim_{Path}$    | 0.4985                | 0.3342          | -0.0003                | 0.4370                | 0.4538                  |
| $Sim_{Wup}$     | 0.4550                | 0.3589          | 0.0252                 | 0.4137                | 0.4080                  |
| $Sim_{Lch}$     | 0.4960                | 0.3544          | 0.0086                 | 0.4097                | 0.4493                  |
| $Sim_{Resnik}$  | 0.4168                | 0.3610          | 0.0539                 | 0.3595                | 0.4471                  |
| $Sim_{Jcn}$     | 0.4823                | 0.3343          | 0.0019                 | 0.4574                | 0.4629                  |
| $Sim_{Lin}$     | 0.4931                | 0.3338          | 0.0147                 | 0.4047                | 0.4712                  |
| WebJaccard      | 0.3272                | 0.4516          | 0.1503                 | 0.0871                | 0.0021                  |
| WebOverlap      | 0.2346                | 0.3953          | 0.0416                 | 0.0778                | 0.0235                  |
| WebDice         | 0.3351                | 0.4365          | 0.1610                 | 0.0871                | 0.0010                  |
| WebPMI          | 0.3272                | 0.4316          | 0.1503                 | 0.0871                | 0.0021                  |
| NGD             | 0.3282                | 0.4671          | -0.2297                | 0.1592                | -0.0446                 |
| LSA-Wikipedia   |                       |                 |                        |                       |                         |
| LDA-Wikipedia   |                       |                 |                        |                       |                         |
| Word2Vec        | 0.6713                | 0.7321          | 0.4527                 | 0.4420                | 0.3635                  |
| FastText        | 0.7529                | 0.8362          | 0.5713                 | 0.4644                | 0.3649                  |
| GloVe           | 0.7152                | 0.8016          | 0.4512                 | 0.4083                | 0.2832                  |
| ELMo            |                       |                 |                        |                       |                         |
| BERT(Embedding) | 0.0019                | 0.0668          | 0.2021                 | 0.0801                | 0.0487                  |

