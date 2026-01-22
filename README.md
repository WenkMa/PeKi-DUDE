# PeKi-DUDE

本项目旨在实现针对 **DUDE** ([Tasks - Document UnderstanDing of Everything 😎 - Robust Reading Competition](https://rrc.cvc.uab.es/?ch=23&com=tasks)) 数据集中测试集的端到端推理流程。

## 🚀 核心架构与方法

本项目采用了一种两阶段的视觉文档处理方案：

- **视觉文档检索 (Visual Document Retrieval):** 使用 **ColQwen2** 进行高效的文档检索。
- **视觉文档问答 (Visual Document QA):** 采用 **Qwen3-VL-2B** 进行多页文档问答推理。
- **参数设置:** 使用默认参数，运行时请注释掉其中分辨率限制(01/22/2026)。

> 更多实现细节请参考源代码。

## 📂 数据说明

**注意：** 为了方便复现，我已经完成了原始测试集格式的转换。

- 处理好的 `.jsonl` 文件已包含在文件夹中。
- 您也可以选择直接从原始问答对数据中自行提取和转换。

## 🤝 贡献与扩展 (Contributing)

这是一个开放的项目，欢迎大家在此基础上进行优化和扩展，例如：

- **重新微调 (Re-finetune):** 在特定数据上重新微调 ColQwen2。
- **模块替换:** 提出并集成新的视觉文档检索模块。
- **模型训练:** 训练针对单页文档问答的新模型。

请自由发挥您的创造力！

## ⭐ Star History

如果您觉得这个工作对您有帮助或有启发，请点亮右上角的 **Star** 支持一下，谢谢！

## 🙏 致谢

非常感谢**ColQwen2**(https://github.com/illuin-tech/colpali) 和 **Qwen3-VL-2B**([Qwen/Qwen3-VL-2B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) 的开源工作为本项目提供了强大的基础模型。

------

### English Version

# PeKi-DUDE

This project implements an inference pipeline specifically designed for the test set of the **DUDE** ([Tasks - Document UnderstanDing of Everything 😎 - Robust Reading Competition](https://rrc.cvc.uab.es/?ch=23&com=tasks)) dataset.

## 🚀 Core Architecture

We employ a two-stage approach for visual document understanding:

- **Visual Document Retrieval (VDR):** Utilizes **ColQwen2** for robust visual retrieval.
- **Visual Document QA (VDQA):** Leverages **Qwen3-VL-2B** for answering questions across multi-page documents.
- **Configuration:** Default conf setting.

> Please refer to the source code for more detailed implementation specifics.

## 📂 Data Preparation

**Note:** To facilitate reproduction, I have pre-processed the original test set.

- The converted format (JSONL files) is included in the directory.
- Alternatively, you can extract and convert the data directly from the original QA pairs if preferred.

## 🤝 Contributing & Future Work

You are welcome to build upon this work. Feel free to explore directions such as:

- **Re-finetuning:** Fine-tune ColQwen2 on domain-specific data.
- **New Modules:** Propose or integrate novel visual document retrieval modules.
- **Model Training:** Train a new model specialized for single-page document QA.

Feel free to innovate and experiment!

## ⭐ Support

If you find this work interesting or helpful, please consider giving it a **Star**.

## 🙏 Acknowledgements

Special thanks to the open-source contributions of **ColQwen2**(https://github.com/illuin-tech/colpali) and **Qwen3-VL-2B**([Qwen/Qwen3-VL-2B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)), which served as the foundation for this project.
