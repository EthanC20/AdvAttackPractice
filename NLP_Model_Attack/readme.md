# 项目描述：

## 题目描述:
* 在这个赛题中，我们将提供一个预训练好的`positive、negative、neutral`文本识别模型以及一组`包含这三类的文本数据集`。

* **参赛选手的任务是：** 对这些文本增加`微小`的扰动，生成攻击文本，使得模型在预测这些经过扰动的攻击文本时出现预测错误。

## 具体要求如下:

* 选手需要设计一种算法，在不明显改变文本语义的前提下，对文本添加微小扰动（同义词替换或者其他方式），使得提供的三分类`positive、negative、neutral`文本识别模型在预测`扰动文本`时出现错误。例如，将原本能够正确识别为`positive`的文本进行扰动后，`模型`会将其错误地分类为`非positive的类别`。

* 为了量化扰动的程度，我们将使用修改后的攻击文本与原始文本的相似度作为评判标准。我们会计算每个扰动文本与其对应原始文本的`语义相似性`。如果修改后的扰动文本与原文本的` 相似度低于75% `，则认为扰动过大，该攻击样本将不被视为有效的对抗性样本。

* 对于满足相似度条件的图像，我们将使用提供的识别模型进行预测。如果可以成功欺骗模型，使其输出错误的模型预测结果，则视为一次成功的攻击，选手将获得 1 分。选手需要在成功攻击至少` 90% `的样本，即产生至少` 90% `满足相似度条件且能够成功欺骗模型的对抗性样本，才能获得题目的 `flag`。


## 项目细节：

1. **相似性的计算**：

    - 相似性的计算流程：

        - 原始文本和修改后的文本被输入到 tokenizer 中，进行编码。编码的结果是将文本转换为模型可接受的输入形式（即张量），并且对这些输入进行填充和截断以确保统一的长度。

        - 代码分别对原始文本和修改后的文本通过模型进行前向传播（forward pass），获取它们的隐藏状态。隐藏状态是由模型内部特定层（在这里是最后一层）生成的向量表示，用于表示输入文本的语义信息。隐藏状态的向量经过平均池化（mean pooling），压缩成一个定长的向量。

        - 使用余弦相似度衡量这两个隐藏状态向量之间的相似性，如果满足相似性要求，则视为通过相似度校验。

    - 参考代码：

        ``` python
        def verify_similarity(original, modified, model, tokenizer):
            # 确保模型处于评估模式
            model.eval()

            # 对原始文本和修改后的文本进行编码
            original_encoding = tokenizer(original, return_tensors='pt', padding=True, truncation=True, max_length=512)
            modified_encoding = tokenizer(modified, return_tensors='pt', padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                # 获取原始文本的隐藏状态
                original_outputs = model.distilbert(**original_encoding)
                original_hidden_state = original_outputs.last_hidden_state.mean(dim=1)

                # 获取修改后文本的隐藏状态
                modified_outputs = model.distilbert(**modified_encoding)
                modified_hidden_state = modified_outputs.last_hidden_state.mean(dim=1)

            # 计算余弦相似度
            similarity = cosine_similarity(original_hidden_state.cpu().numpy(),
                                        modified_hidden_state.cpu().numpy())[0][0]

            return similarity

        ```

2. **数据和模型**：

   - 发放的数据中包含文本文件`original_text.csv`和 `预训练模型 Sentiment_classification_model`。

    - 模型的 `label` 映射: ` {'positive':2,'negative':0,'neutral':1}`

    - 可以使用 `transformers` 模块中的 `AutoTokenizer`、`AutoModelForSequenceClassification` 分别加载 `tokenizer` 和 `model`。

3. **提交要求**：

    - 选手需要把增加扰动后得到的新文本(攻击文本)，按照规定的格式进行上传提交（csv文件），格式如下，保留原id列：
    
        | id  | attacked_text                                                                                              |
        |-----|------------------------------------------------------------------------------------------------------------|
        | 0 | \#powerblog What is this powerblog challenge you keep talking about?  I`m a newbie follower                 |
        | 1 | Good mornin. Today will end early, woo. Gonna work on rick`s surprise PROJECT DUE ON TUESDAY               |


## 评分标准：

1. **相似度**：
   - 生成的对抗文本和原始文本之间的相似度需要在 75% 以内。
   
2. **攻击成功率**：
   - 攻击成功率达到 90%（即生成的对抗样本中有 90% 能够欺骗模型并满足相似度要求）。

3. **成功满足上述条件即可获取比赛的 flag。**
