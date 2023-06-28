# Attention Is All You Need: The Transformer


# Introduction

The advent of deep learning over the past few years has opened a lot of possibilities regarding neural machine translation (NMT). **Attention is all you** **need**, also known as **Transformer** [1], has become the state-of-the-art in NMT, surpassing tradition recurrent neural network (RNN) based encoder-decoder architecture. A bunch of new architectures is now being built based on the transformer. I will discuss how NMT has evolved throughout the last couple of years, from the traditional RNN to the Transformer.

# Terminology

Before describing, how NMT has evolved through different phase shifts, I want to briefly define the terminologies which have been used throughout this article.

## Deep Learning

Deep learning is a sub-field of machine learning which is inspired by the structure and functions of the human brain. Typically, it has some inputs, outputs, and several hidden layers to perceive, process and deliver accordingly. In other words, deep learning is nothing but a group neural network algorithms which can imitate human learning style to some extent i.e. understanding patterns, recognize different persons, objects etc. Instead of neural electric signals, it uses numerical data converted from images, videos, texts, etc from the real world.

We can visualize how data is pipe-lined through such architecture from figure 1.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560119353061.png)

## Natural Language Process (NLP)

Language that we, as human, speak and write as our natural way of communication is called natural language, and when a computer, as a machine, needs that language to be interpreted for accomplishing any task, it processes the language for its own understanding and analysis employing algorithms, then we call such processing as natural language processing.

NLP tasks mostly comprise of two major things, natural language understanding, and natural language generation. Speech recognition, topic modeling, sentiment analysis, translating human speech or writing from one language to another, and generating or imitating Shakespeare’s novel are few of the tasks
that NLP has been dealing with.

In figure 2, we can see a usage of natural language processing where the machine is trying to extract the user’s intention and important entities from the text.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560119481271.png)

## Machine Translation

Using natural language processing, nowadays the computer is able to translate text or speech from a source language (i.e. German) to a target language (i.e. English). Automatic translation by machine is called machine translation (MT).

We always expect the translated speech to be retained completely the same meaning as conveyed in the source language. As we all know, translating word by word, mostly doesn't make any sense. Each language has its own grammar style.

Moreover, people from a particular language don’t always maintain grammar, they can understand their aberrant use of language based on context only. So, the translator, whoever it is, either human or machine, needs to interpret and analyze all of the surrounding and the usage pattern of those words in different contexts along with expertise in grammar, syntax, semantics of the languages involved in translation. Machine translation algorithms can be categorized into two major systems: rule-based machine translation (RBMT) and Statistical machine translation (SMT).

RBMT systems are formed on massive dictionaries and complex linguistic rules. Using these complex rule sets, it transfers the grammatical structure of the source language into the target language.

On the other hand, SMT is built on the analysis of statistical translation models generated from monolingual and bilingual training data. Basically, this system completely relies on the data supplied to its algorithm for learning or generating the complex rules for its model.

Since SMT algorithms take much less time than RBMT and can imitate the pattern of training data to generate target output, SMT technology is the clear winner in the area of machine translation.

**Neural Machine Translation** Algorithms based on neural network, that learn a statistical model for machine translation is called neural machine translation (NMT). The pipeline of specialized systems used in statistical translation is no more needed in NMT. “Unlike the traditional phrase-based translation system which consists of many small sub-components that are tuned separately, neural machine translation attempts to build and train a single, large neural network that reads a sentence and outputs a correct translation” [4]. As such, NMT systems are called end-to-end modeling for machine translation.

**Deep Neural Machine Translation** It is an addition of Neural Machine Translation. Unlike the traditional NMT, deep NMT processes multiple neural network layers instead of just one. As a result, we experienced the best machine translation quality ever produced before it.

Figure 3 shows how Google Translate [5] is helping people translate text from one language to another.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560119717161.png)

# Deep Learning in Machine Translation

In this section, I would like to discuss how machine translation evolved from the very beginning of deep learning until Transformer takes place and gains superiority among all of those algorithms.

## Recurrent Neural Networks in NMT

Recurrent Neural Networks (RNNs) are popular models that have exhibited excellent promise in numerous NLP tasks. Mostly RNNs are used in sequential data i.e. text, audio, signals etc. So, the idea behind RNNs is to make use of sequential information.

In a conventional neural network, each input is independent of other inputs. But, tasks involving sequential dependency cannot be solved assuming such independence. For example, If we want to predict the next word in a sentence, we better know which words came before it. Being said that, there comes the concept of recurrent neural networks where are they perform the same task for every element of a sequence, with the output being dependent on the previous computations or outputs.

In other words, RNNs have “memory” cells which gain information about what has been seen so far. Theoretically, RNNs can make use of information in arbitrarily long sequences, but in practice, they are limited to looking back only a few steps because of time and memory limitations. Figure 4 shows a typical
RNN architecture.

In terms of machine translation, more often we need to process long-term information which has to sequentially travel through all cells before getting to the present processing cell. This means it can be easily corrupted by being multiplied so many times by small numbers. This is the cause of vanishing gradients.

RNNs have difficulties learning long-range dependencies and interactions between words that are several steps apart. That’s problematic because the meaning of an English sentence is often determined by words that aren't very close: “The man who wore a wig on his head went inside”. The sentence is really about a man going inside, not about the wig. But it’s unlikely that a plain RNN would be able to capture such information.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560119869875.png)

## Gated RNNs in NMT

Gated recurrent networks such as those composed of Long Short-Term Memory (LSTM) nodes are other improved versions of RNNs which have been called state-of-the-art in many supervised sequential processing tasks such as speech recognition and machine translation for quite a long time until the other advanced versions have taken place.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560119923053.png)

As we already know, one major shortcoming of RNNs is not being efficient to maintain contexts for longer time sequences. The memory cell of an RNN is updated at each time step with new feedforward inputs. This means that the network does not have control of what and how much context to maintain over
time. Another reason behind this problem is that when RNNs are trained with backpropagation through time, they are not able to properly assign gradients to previous time steps due to squashing non-linearities. This is called the vanishing gradient problem.

LSTMs were designed to overcome the vanishing gradient problem by controlling gradient flow using extra control logic and by providing extra linear path-ways to transfer gradient without squashing. [9]

Gated Recurrent Unit (GRU) is another kind of approach having the same goal of tracking long-term dependencies effectively while mitigating the vanishing/exploding gradient problems. Figures 5 and 6 show basic LSTM and GRU units respectively.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560119983859.png)

These kinds of approaches have contributed a lot in the area of sequence encoding and helped improving vanishing gradient problem as well as uncertainty about remembering long term dependencies to some extent. These approaches do better when the source and target language maintain almost similar word order. On the other hand, gated neural networks still prohibit parallelization.

Long-range dependency issue is not completely solved for long texts. Since these approaches use an encoder-decoder architecture in which target sequence is generated only from the last encoded vector.

## Convolutional Neural Networks in NMT

Convolutional Neural Networks (CNNs) can solve some of the problems that were not resolved. Layer-wise parallelization is possible due to CNN’s trivial architecture [11]. Each source word can be processed at the same time and does not necessarily depend on the previous words to be translated. In addition to that the algorithm for capturing these dependencies scales in O(n/k) instead of O(n) due to the hierarchical structure. Figure 7 shows how WaveNet is structured.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120062403.png)

On the contrary, CNNs needs left-padding for texts. The complexity of O(n) for ConvS2S and O(nlogn) for ByteNet makes it harder to learn dependencies on distant positions.

## Attention-based RNNS in NMT

The attention mechanism was born to help memorize long source sentences in NMT [4]. Rather than building a single context vector out of the encoder’s last hidden state, the secret sauce invented by attention is to create shortcuts between the context vector and the entire source input. The weights of these shortcut connections are customizable for each output element.

While the context vector has access to the entire input sequence, we don’t need to worry about forgetting. The alignment between the source and target is learned and controlled by the context vector. Decoder attends to different parts of the source sequence at each step of the output generation. Each decoder output depends on a weighted combination of all the input states, not just the last state. And thus, the decoder knows whom to attend more than others (i.e “la Syrie” to “Syria” in figure 8). Figure 8 shows alignment matrix of source and corresponding target sentence.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120148585.png)

Attention comes at a cost, we need to calculate an attention value for each combination of input and output word i.e 50 words input sentence would take 2500 attention values to be calculated. Character level computations and dealing with other sequence modeling tasks would be expensive. Counter-intuitive compare to the human attention analogy. It scans all possible details before deciding which is memory intensive and also computationally expensive.

# Transformer

The transformer reduces the number of sequential operations using a multi-head attention mechanism. It also eliminates the recurrence/convolution completely with attention and totally relied on self-attention based auto-regressive encoder-decoder, i.e. uses previously generated symbols as extra input while generating
next symbol -

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120257793.png)

## Model Architecture

We can divide the model into two sub-models, namely encoder, and decoder. Figure 9 shows the model architecture of the Transformer.

### Encoder

1. The encoder is composed of a stack of N= 6 identical layers.
2. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
3. There is a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer _isLayerNorm(x+Sublayer(x))_, where _Sublayer(x)_ is the function implemented by the sublayer itself.
4. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension `$d_{model}$`= 512.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120536027.png)

### Decoder

1. The decoder is also composed of a stack of N= 6 identical layers.
2. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.
3. Similar to the encoder, there are residual connections around each of the sub-layers, followed by layer normalization.
4. The self-attention sub-layer is modified in the decoder stack to prevent positions from attending to subsequent positions.
5. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for the position _i_ can depend only on the known outputs at positions less than _i_.

**Self-attention**: Self-attention, also known as intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the same sequence. It has been shown to be very useful in machine reading, abstractive summarization, or image description generation.

**Multi-Head Attention**: The multi-head mechanism runs through the scaled dot-product attention multiple times in parallel. The independent attention outputs are simply concatenated and linearly transformed into the expected dimensions. According to the paper, “Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.” Figure 10 shows, how multi-head attention works in the Transformer.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120733085.png)

**Masked Multi-Head Attention**: This layer is from the decoder to prevent future words to be part of the attention i.e. at inference time, the decoder would not know about the future outputs. It zeroes-out the similarities between words and the words that appear after the source words (”in the future”). Simply removing such information, the only similarity to the preceding words is considered.

**Positional Encoding**: Another important step on the Transformer is to add positional encoding when encoding each word. Encoding the position of each word is relevant, since the position of each word is relevant to the translation. Since the architecture has completely got rid of recurrence or convolutional sections, a positional encoding is required. Figure 11 shows the equation used for positional encoding in the Transformer.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120815181.png)

The overall flow of data from encoder’s input layers to the decoder’s output layers can be visualized by an unfolded network architecture shown in figure 12.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120889655.png)

# Evaluation of the Transformer

Authors conducted experiments on two machine translation tasks and claim that the Transformer models are superior in quality in terms of parallelization and less training runtime.

Authors say, “The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.” They also show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

Figure 13 and 14 show the comparisons with other SOTA models in terms of BLEU scores, training runtime and FLOPS.

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120956406.png)

![img](https://pub-b4d7e64fdefe48ffbac008dbd2c3c167.r2.dev/attention-is-all-you-need/1560120986077.png)

# Conclusion

The Transformer is an advanced approach for machine translation using attention based layers. It introduces a completely new type of architecture namely self-attention layers. The model gets rid of convolutional or recurrent layers and still achieves state of the art on WMT14 English-German and English-French data sets. The model uses parallel attention layers whose outputs are concatenated and then fed to a feed-forward position-wise layer. The model is not only a great success in machine translation, but also created scopes to improve other NLP tasks. BERT [14] is one of the most influential successors of this model which can create a language model adopting transformer architecture. Neural Speech Synthesis with Transformer Network [15] is another achievement in text-to-speech task that exploits the power of the Transformer model. “The transformer is one of the most promising structures, which can leverage the self-attention mechanism to capture the semantic dependency from the global view. However, it cannot distinguish the relative position of different tokens very well, such as the tokens located at the left or right of the current token, and cannot focus on the local information around the current token either.”, pointed out by the authors of the Hybrid Self-Attention Network (HySAN) [16] that aims to alleviate these problems.

# References

1. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser,
   and I. Polosukhin, “Attention is all you need,” 06 2017.
2. [Online]. Available: https://towardsdatascience.com/applied-deep-learning-part-
   1-artificial-neural-networks-d7834f67a4f6
3. [Online]. Available: https://medium.com/@dc.aihub/natural-language-processing-
   d63d1953a439
4. D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly
   learning to align and translate,” CoRR, vol. abs/1409.0473, 2014. [Online].
   Available: http://arxiv.org/abs/1409.0473
5. “Google translate.” [Online]. Available: https://translate.google.com
6. [Online]. Available: https://hub.packtpub.com/create-an-rnn-based-python-
   machine-translation-system-tutorial
7. W. Feng, N. Guan, Y. Li, X. Zhang, and Z. Luo, “Audio visual speech recognition
   with multimodal recurrent neural networks,” 05 2017, pp. 681–688.
8. [Online]. Available: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
9. R. Józefowicz, O. Vinyals, M. Schuster, N. Shazeer, and Y. Wu, “Exploring
   the limits of language modeling,” CoRR, vol. abs/1602.02410, 2016. [Online].
   Available: http://arxiv.org/abs/1602.02410
10. [Online]. Available: https://en.wikipedia.org/wiki/Gated recurrent unit
11. J. Gehring, M. Auli, D. Grangier, D. Yarats, and Y. N. Dauphin, “Convolutional
    sequence to sequence learning,” CoRR, vol. abs/1705.03122, 2017. [Online]. Available:
    http://arxiv.org/abs/1705.03122
12. [Online]. Available: https://deepmind.com/blog/wavenet-generative-model-raw-
    audio/
13. [Online]. Available: https://research.jetbrains.org/files/material/5ace635c03259.pdf
14. J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep
    bidirectional transformers for language understanding,” CoRR, vol. abs/1810.04805,
15. [Online]. Available: http://arxiv.org/abs/1810.04805
16. N. Li, S. Liu, Y. Liu, S. Zhao, M. Liu, and M. Zhou, “Close to human quality
    TTS with transformer,” CoRR, vol. abs/1809.08895, 2018. [Online]. Available:
    http://arxiv.org/abs/1809.08895
17. K. Song, T. Xu, F. Peng, and J. Lu, “Hybrid self-attention network for
    machine translation,” CoRR, vol. abs/1811.00253, 2018. [Online]. Available:
    http://arxiv.org/abs/1811.00253

