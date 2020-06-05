# DuReader系列数据集
DuReader系列数据集是百度开源的一系列面向机器阅读理解领域的中文数据集，包括**DuReader 2.0** (当前规模最大的开源中文阅读理解数据集)，**DuReader<sub>robust</sub>** (考察阅读理解模型鲁棒性的测试集) 以及**DuReader<sub>yesno</sub>** (分类式观点阅读理解数据集)。

| 数据集名称      |  任务形式   | 数据集规模| 数据集下载 | 基线系统 |
| ----                      | ----       | :----: | :----: | :----: |
| DuReader 2.0              | 多篇章 抽取式| 300K | [link]()| [BiDAF](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2018-DuReader) |
| DuReader<sub>robust</sub> | 单篇章 抽取式| 20K  | [link]()| [ERNIE1.0](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Robust-BASELINE) |
| DuReader<sub>yesno</sub>  | 单篇章 分类式| 85K  | [link]()| [ERNIE1.0](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Yesno-BASELINE) |


## 机器阅读理解任务简介
机器阅读理解 (Machine Reading Comprehension) 是指让机器阅读文本，然后回答和阅读内容相关的问题。其技术可以使计算机具备从文本数据中获取知识并回答问题的能力，是构建通用人工智能的关键技术之一。简单来说，就是根据给定材料和问题，让机器给出正确答案。阅读理解是自然语言处理和人工智能领域的重要前沿课题，对于提升机器智能水平、使机器具有持续知识获取能力具有重要价值，近年来受到学术界和工业界的广泛关注。

## [DuReader 2.0](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2018-DuReader)

### DuReader 2.0数据集
DuReader 2.0是一个大规模、面向真实应用、由人类生成的中文阅读理解数据集。DuReader聚焦于真实世界中的不限定领域的问答任务。相较于其他阅读理解数据集，DuReader的优势包括:

 - 问题来自于真实的搜索日志
 - 文章内容来自于真实网页
 - 答案由人类生成
 - 面向真实应用场景
 - 标注更加丰富细致
 
更多关于DuReader 2.0数据集的详细信息可在[DuReader官网](https://ai.baidu.com//broad/subordinate?dataset=dureader)以及[论文](https://arxiv.org/abs/1711.05073)中找到。

### DuReader基线系统

DuReader基线系统利用[PaddlePaddle](http://paddlepaddle.org)深度学习框架，针对**DuReader 2.0阅读理解数据集**实现并升级了一个经典的阅读理解模型 —— BiDAF. 

**该数据集以及基线系统被用于举办[2018机器阅读理解技术竞赛](http://mrc2018.cipsc.org.cn/)以及[2019语言与智能技术竞赛](http://lic2019.ccf.org.cn/read)。**

## [DuReader<sub>robust</sub>](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Robust-BASELINE)

### DuReader<sub>robust</sub>数据集
阅读理解模型的鲁棒性是衡量该技术能否在实际应用中大规模落地的重要指标之一。随着当前技术的进步，模型虽然能够在一些阅读理解测试集上取得较好的性能，但在实际应用中，这些模型所表现出的鲁棒性仍然难以令人满意。DuReader<sub>robust</sub>数据集作为首个关注阅读理解模型鲁棒性的中文数据集，旨在考察模型在真实应用场景中的鲁棒性，包括

- 过敏感性：问题语义相同，变换问法时，模型预测的答案发生大幅变化
- 过稳定性：模型过分依赖字面匹配，忽略深层次语义信息
- 泛化能力：训练数据的领域和实际应用分布不同时，模型能否表现良好

更多关于DuReader<sub>robust</sub>数据集的详细信息可在[论文](https://arxiv.org/abs/1711.05073)中找到。
### DuReader<sub>robust</sub>基线系统
DuReader<sub>robust</sub>基线系统利用PaddlePaddle深度学习框架以及[ERNIE 1.0](https://arxiv.org/abs/1904.09223)预训练模型，针对DuReader<sub>robust</sub>数据集实现的阅读理解基线系统。

**该数据集以及基线系统被用于举办[2020语言与智能技术竞赛](https://aistudio.baidu.com/aistudio/competition/detail/28?isFromCcf=true)。**

## [DuReader<sub>yesno</sub>](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Yesno-BASELINE)

### DuReader<sub>yesno</sub>数据集
机器阅读理解评测中常用的F1、EM等指标虽然能够很好的衡量抽取式模型所预测的答案和真实答案的匹配程度，但在处理观点类问题时，该类指标难以衡量模型是否真正理解答案所代表的含义，例如答案中包含的观点极性。DuReader<sub>yesno</sub>是一个以观点极性判断为目标任务的数据集，通过引入该数据集，可以弥补抽取类数据集的不足，从而更好地评价模型的自然语言理解能力。
### DuReader<sub>yesno</sub>基线系统
DuReader<sub>yesno</sub>基线系统利用PaddlePaddle深度学习框架以及[ERNIE 1.0](https://arxiv.org/abs/1904.09223)预训练模型，针对DuReader<sub>yesno</sub>数据集实现的**分类式**阅读理解基线系统。

**该数据集以及基线系统被用于举办[2020中国人工智能大赛](https://ai.ixm.gov.cn/2020/index.html)。**