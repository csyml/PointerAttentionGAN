# PointerAttentionGAN
##1.思路
<!--* 蛋白序列设计完全可以理解为文章摘要撰写的问题：
  - 需要提取原序列中的关键词的同时生成纳入原序列中全部字符空间中***相关字符***从而形成有有意义的序列。
  - 关键字相当于功能蛋白序列中的核心功能子片段，***相关字符***可以对应所有可能氨基酸。
   
##2.实现
* 在ProteinGAN的生成器和判别器上直接并行Pointer-Generator-Coverage，最后对输出向量进行叠加；
* Pgen来作为软选择的概率。决定当前预测是直接从源文本中复制一个词过来还是从词汇表中生成一个词出来。
* Coverage机制来解决重复问题。如果模型在预测时总是注意相同的部分，那么就很有可能会预测出相同的单词，
  因此为了防止这种情况发生，Coverage机制强迫模型多去关注之前没被注意过的角落。
* TransGAN 在multi-attention 并行执行pointer-network与coverage,Pgen。
* RNN 如何与GNA无缝连接。

##3.解决蛋白序列设计中的问题
* pointer-network 直接提取高性能力学蛋白序列中的功能片段；
* 解决蛋白序列长度定长问题；
* 解决蛋白序列重复子序列问题问题；
* 准确描述蛋白序列中子序列相关性问题。

##4.Reference
* Language Generation with Recurrent Generative Adversarial Networks
without Pre-training
* Bidirectional Conditional Generative Adversarial 
* Multi-Generator Generative Adversarial Nets
* Multi-Generator Generative Adversarial Nets
* Bidirectional Conditional Generative Adversarial Networks
* Variational Approaches for Auto-Encoding Generative Adversarial Networks
-->
