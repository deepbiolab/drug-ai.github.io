---
layout:     post
title:      "AI for Drug Discovery"
subtitle:   "Review of Deep Learning in Pharmaceutic"
date:       2020-11-10 12:00:00
author:     "ShaneTin"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Drug Discovery
---



### 前言

深度学习算法在很多领域已经可以成功落地，并且效果很好，例如以卷积神经网络为例，该算法以及衍生的系列算法在图像处理领域效果甚佳，例如图像分类，目标检测等，而以循环神经网络为代表的算法在序列处理领域有非常好的效果，例如在机器翻译，语音识别，以及文本生成，音乐生成方面都取得了很大的进步。

对于药物发现这个领域而言，很多时候对于药物开发是一个非常耗时以及繁琐的过程，而且中间充满着各种不确定性，试错成本和代价也很大，如果能够通过深度学习的相关技术去操纵分子数据并从中发现见解，或者进行药物研发环节中的各种模拟，将很大程度上缩减时间周期以及试错成本。例如分子数据最常见的表示方式，一种是以类似分子式的序列表示方式，可以很好的对其进行序列建模，而深度学习中的经典的RNN模型就可以解决类似的问题，包括近年来的seq2seq或者transformer架构都给这项任务提供了丰富的操作空间。而另一种，以图形方式表示的分子数据，则可以借用CNN系的算法去同样解决类似的问题。

本人在药物研究所的工作经历，以及国内外近年来深度学习在各个领域的应用，让我相信，药物研发借助于AI去助力是必经之路， 这也是我开这个博客的初衷之一。 另外大学时期在医学工程学院的药物研发知识以及毕业后从事医药与数据挖掘交叉领域的工作，使我逐渐认识到算法的美妙和医药对于当前社会的重要性。因此我想通过这个博客，去记录我个人对于这两个领域融合的所见所想，以及分享相关该领域的最前沿的进展，一方面，主要是个人工作的汇总，另一方面也会通过体系的教程去帮助到其他对该领域感兴趣的人。

关于这篇综述，我会从三个核心方面去进行介绍，后期，我会对每一块内容进行更详细的介绍。那么关于深度学习在药物发现领域的应用我认为主要包括以下三个部分。

- 分子的表示方式
- 药物与靶标的相互作用预测
- 新药设计



### 分子表示方式

机器学习问题大体上分为三类，监督学习，半监督学习，强化学习。例如以药物性质预测任务来讲，我们可以把它划分一个二分类的监督学习任务。

那么此时对于模型的输出，只有两种结果（0，1），也即是否该药物具有某种性质。有则为1，无则为0。

而对于模型的输入，则有多种表示方式。对于机器学习算法而言，如果该特征本身是数值变量那么可以使用它本身作为输入，对于类别变量而言，最直接的方式便是通过one-hot encoding的方式进行表示，那么同样的，对于一个化合物分子，不管是大分子还是小分子，其均有相应的结构与之依附，那么对这些结构的不同表示方式，也就决定了模型的特征表示方式。总体主要包括如下四个分类。

#### Fingerprint
其中表示药物的一种方法是分子指纹。 指纹的最普遍类型是一系列二进制数字（位），代表分子中是否存在特定的子结构。 因此，药物（小化合物）被描述为0和1的向量（数组）。如下图所示：

![如何以二进制向量表示分子](https://tva1.sinaimg.cn/large/0081Kckwgy1gkmg9dq248j31540memxn.jpg)

这种表示方式的优点是简单快速，而且也在文献中被广泛使用[<sup>1</sup>](#refer-anchor)。 但是，很明显，将分子编码为二进制向量不是一个可逆的过程（这是有损的转化）。 即，我们可以将一个能够表示结构信息的分子式编码成分子指纹，但是却不可以从分子指纹中推断出该分子有怎样的结构。

表示一个小分子可以有很多不同的指纹。 可以按照RDKit官方文档[2]进一步了解它们。



#### SMILES

表示分子的另一种方法是将结构编码为文本。 这是
将图结构数据转换为文本内容，并在机器学习输入管道中使用文本（编码字符串）作为输入。  Simplified Molecular-Input Line-Entry System（SMILES）是标准和最受欢迎的表示之一。 转换后，我们可以使用自然语言处理（NLP）的相关算法来处理药物，例如，预测其性质，副作用甚至化合物之间的相互作用 [3]。

![如何用SMILES码来表示分子](https://tva1.sinaimg.cn/large/0081Kckwgy1gkmgyob13hj30vu0u0myp.jpg)

有关SMILES的更多信息，可以单击此链接[4](http://opensmiles.org/opensmiles.html)。



#### InChIKey

尽管SMILES在化学家和机器学习研究人员中非常受欢迎，但它并不是唯一可用于表示药物的基于文本的表示形式。 InChIKey是您可以在文献中找到的另一种流行的表示形式。InChI国际化合物标识是（国际化合物标识）International Chemical Identifier的缩写. InChI编码是一串由斜杠（/）隔开的有层级关系的数字组成的。每个InChI编码都是由InChI版本号开始，接着一个主层号。主层下包括含化学分子式层、原子关系层和固定氢原子子层。基于分子结构的主层后往往接着一个附加的层，如电荷层、立体化学层（和/或）同位素信息层。

> 以维生素C的国际化合物标识码为例
>
> InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1
>
> InChI Key：CIWBSHSKHKDKBQ-JLAZNSOCSA-N



InChIKey 是对 InChI 运用 SHA-256 算法处理后得到的哈希值，它的出现是为了解决 InChI 长度不定的问题。与 InChI 相比，InChIKey 具有这样几个特点：

- 长度固定，永远是27个字母
- 与 InChI 几乎一一对应，但有很小的概率（1/10亿）出现两个 InChi 对应同一个InChIKey
- 不可读，字符串本身没有意义，必须转换回 InChI 才能读
- 在实际使用中，可以用InChIKey 作为关键字检索出对应的 InChI，再做进一步的使用。



#### SELFIES

为了解决SMILES的表示方法有时候不能对应有效的分子， Mario Krenn et al.[5](https://arxiv.org/pdf/1905.13741.pdf) 提出了一种新的分子表示方法， 即SELFIES（SELF-referencIng Embedded Strings），它是基于字符串的表示形式。每个SELFIES字符串都对应一个有效分子。



![image-20201112164754376](https://tva1.sinaimg.cn/large/0081Kckwgy1gkmhjkib2pj31150u0myx.jpg)




#### Graph

深度学习盛行于图结构化数据，例如[图卷积网络](https://tkipf.github.io/graph-convolutional-networks/) [6]使直接使用图数据作为深度学习管道的输入成为可能。
例如，可以将化合物视为图，其中顶点是原子，原子之间的化学键是边。 图神经网络领域中，有专门用于此工作的库，如[Deep Graph Library](https://www.dgl.ai/)，[PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric)，[PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph)





### 药物与靶标的相互作用预测

蛋白质在生物中起着核心作用。即，蛋白质是生物细胞内部和外部大部分功能的关键参与者。例如，有些蛋白质负责细胞凋亡，细胞分化和其他关键功能。同时，蛋白质的功能直接取决于其三维结构。即，改变蛋白质的结构可以显着改变蛋白质的功能，这是药物发现的重要依据之一。许多药物（小分子）被设计与特定蛋白质结合，改变其结构，从而改变其功能。此外，至关重要的一点是，仅改变一种蛋白质的功能就可以对细胞功能产生巨大影响。蛋白质直接彼此相互作用，并且某些蛋白质还充当转录因子，这意味着它们可以抑制或激活细胞中其他基因的表达。因此，改变一种蛋白质的功能可以对细胞产生巨大的影响，并可以改变不同的细胞通路。



那么，药物发现中的一个重要问题是预测特定药物是否可以结合特定蛋白质。而药物-靶标相互作用（DTI）预测任务，近年来受到了极大的关注。



> 我们可以像下面这样构造DTI预测任务：
> 描述：预测化合物与蛋白质结合亲和力的二元分类（可以形式化为回归任务或二元分类）
>
> 输入：化合物和蛋白质表示向量
>
> 输出：0-1或[0-1]中的实数



[Qingyuan Feng](https://arxiv.org/abs/1807.09741)[7]提出了一种基于深度学习的药物-靶标相互作用预测框架。 用于DTI预测的大多数深度学习框架都将化合物和蛋白质信息作为输入，但是不同之处在于它们用于输入神经网络的输入表示方法的不同。 正如我在上一节中提到的，化合物可以多种方式表示（分子指纹，SMILES，从图卷积网络提取的特征），蛋白质也可以具有不同的表示。 根据不同的表示，可以使用各种网络架构来处理DTI预测。
例如，如果我们要对化合物和蛋白质都使用基于文本的表示形式（化合物和氨基酸的SMILES代码或蛋白质的其他基于序列的描述符），那么基于RNN的体系结构就是我想到的第一件事。



[Matthew Ragoza](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740)等。 提出了一种用卷积神经网络进行蛋白质配体评分的方法[8]。 他们没有使用基于文本的表示，而是利用了蛋白质配体的三维（3D）表示。 因此，使用可以作用于此3D结构的卷积神经网络，并提取有意义和适当的特征以预测蛋白质配体结合亲和力。



最近，[Bo Ram Beck](https://www.biorxiv.org/content/10.1101/2020.01.31.929547v1)等人。 使用分子的SMELIES表示并使用用双向编码器表示的transformer架构(BERT), 开发了一种可以筛选对2019-nCoV病毒蛋白起作用的市售药物的模型-Molecule Transformer Drug Target Interaction (MT-DTI)。



尽管提出深度学习算法用于DTI预测已成为一种大趋势，并且在某些情况下已经取得了令人印象深刻的结果，但论文非常相似，而我发现的唯一创新就是选择了输入表示的不同。 因此，对于该项任务总结如下：

- 查找包含有关化合物和目标物以及它们是否相互作用的信息的数据库（例如[STITCH](http://stitch.embl.de/)数据库）
- 在DTI预测中，最常见的网络是将一对化合物和蛋白质作为输入
- 选择适合化合物和蛋白质的表示形式
- 根据选择的表示形式，选择合适的神经网络模型来处理输入。 根据经验，对于输入， 如果是基于文本的表示，可以使用基于RNN的体系结构（GRU，LSTM等）和transformer，对于图像或3D结构，可以使用卷积神经网络。
- 该问题可以看作是二元分类（化合物是否结合到靶标）或回归（预测化合物与蛋白质之间的亲和力强度）。



以上是DTI预测的大致内容。 起初，也许这似乎是一项艰巨的任务，但是借助deep learning算法可以以非常简单的技术和策略来解决这一问题。

> 输入表示：input representation



### 新药设计

到目前为止，我们仅涉及到了判别式算法。 即，给定一种药物，该算法可以预测其副作用和其他相关特性，或者给定化合物-蛋白质对，则可以预测它们是否可以结合。



但是，如果我们对设计具有某些特性的化合物感兴趣呢？ 例如，我们要设计一种化合物，该化合物可以与特定蛋白质结合，修饰某些通路，并且不与其他通路相互作用，并且还具有某些物理性质，例如特定的溶解度范围。



上一部分介绍的工具链是无法解决这个问题的。 这个问题最好在生成模型的领域中实现。自回归算法(Autoregressive)，变分自编码器（VAE）和生成对抗网络（GAN）等生成模型已经在机器学习社区中得到了广泛普及。 但是，在新药物设计任务中的应用还不是很久。



显而易见，产生具有某些所需特性的化合物比上一节中讨论的其他两个问题难。可供搜索的化学分子的空间非常大，在该空间中进行搜索以找到合适的药物非常耗时且几乎是不可能的任务。 尽管有些文献中有一些不错的结果，但该领域尚处于起步阶段，需要更成熟的方法。 在这里，我将回顾我在该领域阅读的一些最佳论文。



很多论文中提及， 生成SMILES作为输出，最后将SMILES转换到化学空间，获取其分子结构。例如，
[Rafael Gomez-Bombarelli](http://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)等， 提出了一种使用数据驱动的分子连续表示进行自动化学设计的方法[9]。



![Using the variational autoencoder to generate compounds with desired properties](https://tva1.sinaimg.cn/large/0081Kckwgy1gkqt60n75bj312c0lgwfx.jpg)





他们使用VAE算法生成分子。 输入表示和输出表示都是SMILES。本文的一个不错的技巧是在潜在空间（它是一个连续空间）中使用高斯过程达到具有所需化学性质的点。然后，使用解码器将潜在空间中的此点转换（解码）为SMILES代码。 该论文写得很好，绝对是推荐读物。但是，问题在于SMILES代码与分子之间没有一一对应的关系。 也就是说，并非所有产生的代码都可以转换回原始（化学）空间，因此，产生的SMILES代码通常与有效分子符。



SMILES是非常流行的表示形式，但它们也具有一个很大的缺点：SMILES并不是可靠的表示形式。 即，更改SMILES中的一个字符（字符突变）可以将分子从有效更改为无效。



[Matt J. Kusner](https://arxiv.org/pdf/1703.01925.pdf)等。通过Grammer VAE专门解决上述提到的“产生与有效分子不对应的SMILES代码问题”[10]。
他们没有将SMILES字符串直接输入到网络并生成SMILES代码，而是将SMILES代码转换为解析树（通过使用SMILES上下文无关的语法）。 使用语法，它们可以生成语法上更有效的分子。 此外，作者指出：

> Surprisingly, we show that not only does our model more often generate valid outputs, it also learns a more coherent latent space in which nearby points decode to similar discrete outputs.



最近，Mario Krenn等人。 提出了另一种基于VAE和SELFIES表示的分子生成方法[5]。 SELFIES的主要优点是坚固性。



下图主要涵盖了使用不同的分子表示方法，以及不同的生成算法的相关论文研究。

![method](https://tva1.sinaimg.cn/large/0081Kckwgy1gkqtibxwelj30jg0audge.jpg)



### 总结

在这篇综述中，回顾了深度学习在药物发现中的一些应用。 显然，这篇评论还没有完成，我会在后续继续补充更多的内容。我希望这篇文章能鼓励你为该领域做出贡献，以使药物发现的工作多一点方便，少一点乏味。



### 引用

<div id="refer-anchor"></div>

[1] [Database fingerprint (DFP): an approach to represent molecular databases](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0195-1%23Sec11)

[2] [Fingerprints in the RDKit](https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf)

[3] [DeepCCI: End-to-end Deep Learning for Chemical-Chemical Interaction Prediction](https://arxiv.org/abs/1704.08432)

[4] [OpenSMILES specification.](http://opensmiles.org/opensmiles.html)

[5] [SELFIES: a robust representation of semantically constrained graphs with an example application in chemistry, Mario Krenn et al.](https://arxiv.org/abs/1905.13741)

[6] [GRAPH CONVOLUTIONAL NETWORKS, Thomas Kipf](https://tkipf.github.io/graph-convolutional-networks/)

[7] [PADME: A Deep Learning-based Framework for Drug-Target Interaction Prediction, Qingyuan Feng et al.](https://arxiv.org/abs/1807.09741)

[8] [Protein-Ligand Scoring with Convolutional Neural Networks, Matthew Ragoza et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740)

[9] [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules, Rafael Gomez-Bombarelli et al.](http://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)

[10] [Grammar Variational Autoencoder, Matt J. Kusner et al.](https://arxiv.org/pdf/1703.01925.pdf)

[11] [Objective-Reinforced Generative Adversarial Networks (ORGAN) for SequenceGeneration Models, Gabriel Guimaraes et al.](https://arxiv.org/pdf/1705.10843.pdf)

[12] [Junction Tree Variational Autoencoder for Molecular Graph Generation, Wengong Jin et al.](https://arxiv.org/abs/1802.04364)

[13] [Deep learning enables rapid identification of potent DDR1 kinase inhibitors, Alex Zhavoronkov et al.](https://www.nature.com/articles/s41587-019-0224-x)

[14] [Augmenting Genetic Algorithms with Deep Neural Networks for Exploring the Chemical Space, AkshatKumar Nigam et al.](https://arxiv.org/abs/1909.11655)

[15] [A Model to Search for Synthesizable Molecules, John Bradshaw et al.](John Bradshaw et al)


