<user>:
到2026.8为止，Describe-Anything 3B这个赛道有没有新的 明显更好的竞品? 即稍小的大模型，详细描述图像中任意物品; sentence-t5这个赛道呢?

<bot>:
截至 2026 年 8 月 3 日，我的判断是：

Describe Anything 3B：出现了一个基本符合“更小、同赛道、明显更强”的直接竞品——GAR-1B。 但它的实际完整权重约 2B，不是字面上的 1B；而且“明显更强”主要体现在多区域关系和推理，单区域详细描述的提升属于稳定但不算碾压。

Sentence-T5：已经被新一代 embedding 模型实质性取代。 不过需要按“检索/RAG”与“通用句子相似度”分别选型。


1. Describe Anything 3B 赛道

DAM-3B 的精确定义是：输入整张图以及点、框、涂画或 mask，结合局部细节和全图上下文，为指定区域生成详细描述；模型约 3B，官方权重仅限非商业用途。

最值得看的竞品：GAR-1B

GAR 支持与 DAM 基本相同的点、框、scribble、mask 输入，不仅描述单个区域，也能同时理解多个区域之间的动作、位置与关系，并回答关于指定区域的自由问题。模型名字叫 GAR-1B，但 Hugging Face safetensors 标注的完整参数量约为 2B，其基础模型则是 Perception-LM-1B。

模型	完整规模	DLC-Bench 单区域描述	其他能力	判断

DAM-3B	3B	64.5／72.6	单区域详细描述、专门的视频时序描述	原基准
GAR-1B	约 2B	67.9／77.1	多区域关系、区域 VQA、组合推理	目前最直接的升级
PLM-SAMTok	1B	63.9	mask 输入与输出、分割能力	更小，但描述未超过 DAM
Qwen3VL-SAMTok	4B	65.6	文档、截图、多面板和分割更强	更通用，但不是更小或更强的 DLC 替代


GAR 在论文给出的两个 DLC-Bench 评测口径下分别比 DAM 高 3.4 和 4.5 分。因此只看“描述一个 mask 中的物体”，它是中等幅度的稳定提升；加入多个物体关系、交互和区域问答后，优势才更明显。不过，后面这些较大的领先主要来自 GAR 作者新建的 GAR-Bench，仍建议用自己的图片分布做盲测。

SAMTok 是 2026 年另一条值得关注的路线：把任意 mask 编码为两个特殊 token，从而让普通 MLLM 同时进行 mask 理解和生成。但在最直接的 DLC-Bench 上，PLM-SAMTok-1B 为 63.9、Qwen3VL-SAMTok-4B 为 65.6，仍低于 DAM-3B 的 67.3；它真正明显占优的是文档、OCR、截图、多面板以及交互式分割，而不是纯粹的任意物体详细描述。

实际选型

静态图片，用户圈出任意物体后要求长而准确的描述：优先 GAR-1B。

涉及多个指定物体的关系或问答：GAR-1B 明显比 DAM 更合适。

视频里需要跟踪目标并描述随时间发生的变化：暂时不建议无条件替换 DAM。 GAR 最突出的公开视频结果主要来自 GAR-8B 的零样本迁移，而 DAM 本身就是针对局部视频时序描述设计的。

还有一个现实限制：DAM 和 GAR 的公开权重都不是宽松商业许可。DAM 是 NVIDIA Noncommercial License，GAR-1B 是 Fair Noncommercial Research License。商业产品不能因为 GAR 性能更好就直接替换上线。

所以这条赛道的结论是：

> 有直接竞品：GAR-1B。它是 2B 对 3B，在静态图像区域描述上小幅到中幅领先，在多区域关系和推理上明显领先；但视频能力及商业许可没有全面解决。




---

2. Sentence-T5 赛道

我这里把 Sentence-T5 理解为“生成固定维度句向量，用于语义相似度、分类、聚类、检索和 RAG”。

原始 ST5 有多个档位：encoder-only 约 110M、335M、1.24B、4.8B；encoder-decoder 约 220M、770M、3B、11B。它主要使用 SentEval、STS 和 SentGLUE，而现在的模型通常报告 MTEB/MMTEB，因此旧分数和新分数不能直接做数值上的横向增益比较。

目前更合理的替代方案

使用场景	推荐模型	参数量	关键特征

小模型多语言检索/RAG	Granite Embedding 97M Multilingual R2	97M	32K 上下文、200+语言、384维、Apache 2.0
通用多语言/中文/RAG	Qwen3-Embedding-0.6B	0.6B	32K、100+语言、32–1024可变维度、Apache 2.0
手机或端侧通用 embedding	EmbeddingGemma-300M	300M	100+语言、128–768维、量化损失较小
开放训练数据、低资源语言和代码	F2LLM-v2-330M	330M	200+语言，模型、数据、代码和中间 checkpoint 开放
英文 CPU 部署	gte-modernbert-base	149M	8K 上下文、English-only、Apache 2.0


最“像小一号 Sentence-T5”的新模型

Granite Embedding 97M Multilingual R2

这是 2026 年最值得注意的紧凑 embedding 模型之一。它只有 97M，比 ST5-Base 的约 110M 还小，支持 32,768 token、200 多种语言，并采用 Apache 2.0 许可。论文报告它在 MTEB-v2 多语言检索平均分达到 60.3，在公开的 100M 以下多语言检索模型中排名最高，并比同规模 multilingual-e5-small 高约 9.4 分。

但它主要是检索专用模型。如果 Sentence-T5 原来用于向量数据库、语义搜索或 RAG，这可以算明确的“小而更好”替代；如果主要用于对称 STS、聚类或特征分类，不能仅凭 Retrieval 分数断言全任务都更好。

Qwen3-Embedding-0.6B

这是目前比较稳妥的通用默认项：0.6B、100 多种语言、32K 上下文，支持任务 instruction 和 32–1024 的可变向量维度，许可证为 Apache 2.0。它特别适合中文/英文混合知识库、长文档检索以及后续搭配 reranker。

它明显小于 ST5 的 1.24B/3B/11B 档，但不小于 ST5-Base 110M 或 ST5-Large 335M，所以不能笼统称为“所有 Sentence-T5 的更小替代”。

EmbeddingGemma-300M 与 F2LLM-v2-330M

EmbeddingGemma 为 300M，略小于 ST5-Large 的 335M，官方结果为 English MTEB v2 69.67、Multilingual MTEB v2 61.15，并支持 128–768 维输出；它更偏成熟的端侧通用方案，但使用 Gemma 许可而非 Apache。

F2LLM-v2-330M 是 2026 年的新竞争者，覆盖 200 多种语言，整个系列从 80M 到 14B，并公开模型、训练数据、代码和中间 checkpoint。它适合重视可复现性、低资源语言和代码 embedding 的团队；不过生态和独立验证程度暂时不如 Qwen3 或 EmbeddingGemma。

极小模型观察项：Bekko

Bekko Embedding 在 2026 年 7 月 28 日才发布。a8m/a25m 分别约有 8M/25M 活跃参数，但包含词表后总参数约为百兆级；支持 8192 token，并报告多语言检索分数 56.2/57.5，可在树莓派和浏览器运行。它很适合极端边缘检索，但发布仅数日、重点又是 retrieval，目前不宜称作通用 Sentence-T5 的成熟赢家。

最终建议

按原 Sentence-T5 的使用方式选择：

原用途	2026 年建议

向量数据库、RAG、语义搜索，模型越小越好	Granite Embedding 97M R2
中文+英文、多任务、长上下文	Qwen3-Embedding-0.6B
端侧且希望保持通用任务能力	EmbeddingGemma-300M
要开放训练数据和可复现训练链	F2LLM-v2-330M
英文、CPU、低延迟	gte-modernbert-base 149M


因此：

> Sentence-T5 赛道的替代比 DAM 更彻底。用于 retrieval/RAG 时，97M 的 Granite R2 已经是很明确的更小换代；用于真正“通用句向量”时，Qwen3-Embedding-0.6B、EmbeddingGemma-300M 和 F2LLM-v2-330M 是更合理的现代候选，但由于 ST5 与它们使用的基准不同，不能严谨地声称每项任务都数值碾压。



