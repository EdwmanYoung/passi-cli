# PassiAgent — Software Requirements Specification

> **版本:** 0.1.0 | **日期:** 2026-06-25 | **状态:** Draft

## 1. 产品概述

### 1.1 产品定位

PassiAgent 是一个面向生信组学下游分析的智能代理（Harness Agent），协助研究人员开展基因组学、表观遗传学、转录组学、代谢组学、蛋白组学以及临床统计学的单组学数据分析和多组学数据联合分析。

### 1.2 目标用户

| 用户角色 | 描述 | 典型场景 |
|---------|------|---------|
| **生物信息学研究员** | 熟悉组学数据，需要高效完成标准化分析 | 差异分析、富集分析、网络分析 |
| **临床研究者** | 有临床数据，需要统计分析和生存分析 | KM曲线、Cox回归、ROC分析 |
| **计算生物学学生** | 学习组学分析方法，需要引导式分析 | 交互式教学、分析流程复现 |
| **多组学项目负责人** | 需要整合多种组学数据发现生物标志物 | MOFA整合、DIABLO分类、WGCNA |

### 1.3 设计参考

架构参考 Kimi CLI 的分层设计模式：Soul 协议、Wire 通信、Runtime 依赖注入、工具优先架构。

---

## 2. 数据范围与格式

### 2.1 基因组学 (Genomics)

| 数据格式 | 扩展名 | 描述 |
|---------|--------|------|
| FASTA | `.fa`, `.fasta`, `.fna` | 参考基因组序列 |
| FASTQ | `.fq`, `.fastq` | 测序读段 + 质量值 |
| SAM/BAM/CRAM | `.sam`, `.bam`, `.cram` | 比对文件 |
| VCF/BCF | `.vcf`, `.bcf`, `.vcf.gz` | 变异检出（SNP, Indel, CNV） |
| GFF3/GTF | `.gff`, `.gff3`, `.gtf` | 基因注释 |
| BED | `.bed` | 基因组区间 |
| PLINK | `.bed/.bim/.fam` | GWAS 基因型数据 |
| MAF | `.maf` | 突变注释格式 |

### 2.2 表观遗传学 (Epigenetics)

| 数据格式 | 扩展名 | 描述 |
|---------|--------|------|
| narrowPeak | `.narrowPeak` | ENCODE 窄峰格式（ChIP-seq/ATAC-seq） |
| broadPeak | `.broadPeak` | ENCODE 宽峰格式 |
| bigWig | `.bigWig`, `.bw` | 覆盖度轨道（二进制索引） |
| bedGraph | `.bedGraph` | 覆盖度/甲基化轨道 |
| Bismark .cov | `.cov` | 单碱基甲基化覆盖度 |
| Hi-C | `.hic`, `.cool`, `.mcool` | 染色质构象 |

### 2.3 转录组学 (Transcriptomics)

| 数据格式 | 扩展名 | 描述 |
|---------|--------|------|
| FASTQ | `.fq`, `.fastq` | RNA-seq 原始读段 |
| BAM/CRAM | `.bam`, `.cram` | 比对读段 |
| Count Matrix | `.csv`, `.tsv` | 基因表达计数矩阵 |
| AnnData | `.h5ad` | 单细胞数据（Scanpy/Seurat） |
| FPKM/RPKM/TPM | `.csv`, `.tsv` | 标准化表达量 |
| GCT/CLS | `.gct`, `.cls` | GSEA 输入格式 |

### 2.4 蛋白组学 (Proteomics)

| 数据格式 | 扩展名 | 描述 |
|---------|--------|------|
| mzML | `.mzML` | HUPO-PSI 标准质谱数据 |
| mzXML | `.mzXML` | 遗留质谱数据格式 |
| MGF | `.mgf` | Mascot 通用格式（峰列表） |
| mzID | `.mzID` | 肽段/蛋白鉴定结果 |
| mzTab | `.mzTab` | 蛋白定量汇总表 |
| PDB | `.pdb` | 蛋白三维结构 |
| 定量矩阵 | `.csv`, `.tsv` | 蛋白表达矩阵 |

### 2.5 代谢组学 (Metabolomics)

| 数据格式 | 扩展名 | 描述 |
|---------|--------|------|
| 丰度矩阵 | `.csv`, `.tsv` | 代谢物丰度表 |
| mzML/mzXML | `.mzML`, `.mzXML` | 质谱原始数据 |
| NetCDF | `.cdf`, `.nc` | GC-MS/LC-MS 数据 |
| ISA-Tab | `.isa.txt` | MetaboLights 元数据 |

### 2.6 临床/表型数据 (Clinical/Phenotype)

| 数据格式 | 扩展名 | 描述 |
|---------|--------|------|
| 临床数据表 | `.csv`, `.tsv`, `.xlsx` | 临床变量、生存数据 |
| CDISC SDTM/ADaM | `.xpt`, `.sas7bdat` | 临床试验数据 |
| REDCap 导出 | `.csv` | 电子 CRF 数据 |

---

## 3. 功能需求

### 3.1 数据管理 (Data Management)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-DATA-01** | 自动检测组学数据格式（40+ 格式支持） | P0 |
| **F-DATA-02** | 读取并预览数据内容（前 N 行、维度、列信息） | P0 |
| **F-DATA-03** | 导出结果为 CSV/TSV/JSON/Excel/HTML 格式 | P0 |
| **F-DATA-04** | 数据格式自动推断（组学类型检测） | P1 |
| **F-DATA-05** | 缺失值检测与插补（KNN、概率最小值、MOFA） | P1 |
| **F-DATA-06** | 数据标准化（log10, CLR, quantile, Z-score, TMM, RLE） | P1 |
| **F-DATA-07** | 批次效应校正（ComBat, Harmony） | P2 |
| **F-DATA-08** | 离群样本检测（PCA, Mahalanobis, IQR） | P2 |

### 3.2 转录组学分析 (Transcriptomics)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-TX-01** | 差异表达分析：DESeq2（R）、edgeR（R）、limma-voom（R）、PyDESeq2（Python） | P0 |
| **F-TX-02** | 基因集富集分析：GSEA（gseapy/fgsea）、ORA（clusterProfiler） | P0 |
| **F-TX-03** | WGCNA 共表达网络分析 | P0 |
| **F-TX-04** | 单细胞分析：聚类、差异表达、拟时序（Scanpy Python / Seurat R） | P1 |
| **F-TX-05** | 火山图、MA图、热图可视化 | P0 |
| **F-TX-06** | GO/KEGG/Reactome 通路注释 | P1 |

### 3.3 基因组学分析 (Genomics)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-GN-01** | GWAS 关联分析（PLINK） | P1 |
| **F-GN-02** | Manhattan 图 / QQ 图 | P1 |
| **F-GN-03** | 变异注释与优先级排序 | P2 |
| **F-GN-04** | CNV 检出与分段分析 | P2 |

### 3.4 表观遗传学分析 (Epigenetics)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-EP-01** | Peak calling QC 指标（FRiP, NSC, RSC） | P1 |
| **F-EP-02** | 差异 Peak/结合分析（DiffBind） | P1 |
| **F-EP-03** | Motif 富集分析（HOMER/MEME） | P2 |
| **F-EP-04** | 差异甲基化区域检出（DSS, bumphunter） | P1 |
| **F-EP-05** | 甲基化 Beta 值分布与可视化 | P1 |

### 3.5 蛋白组学分析 (Proteomics)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-PR-01** | 差异蛋白丰度分析（limma, MSstats） | P1 |
| **F-PR-02** | PTM 分析、蛋白复合体富集 | P2 |
| **F-PR-03** | 通路富集（ReactomePA） | P1 |

### 3.6 代谢组学分析 (Metabolomics)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-MT-01** | 差异代谢物丰度分析 | P1 |
| **F-MT-02** | 峰对齐与鉴定（XCMS/MZmine 风格） | P2 |
| **F-MT-03** | 通路映射（KEGG, MetaCyc） | P1 |

### 3.7 临床统计学 (Clinical Statistics)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-CS-01** | Kaplan-Meier 生存曲线 + log-rank 检验 | P0 |
| **F-CS-02** | Cox 比例风险回归 + 假设检验 | P0 |
| **F-CS-03** | 正则化 Cox（lasso/ridge/elastic net） | P1 |
| **F-CS-04** | 竞争风险模型（Fine-Gray, cause-specific） | P1 |
| **F-CS-05** | RMST（限制平均生存时间） | P2 |
| **F-CS-06** | ROC/AUC、敏感性/特异性分析 | P1 |
| **F-CS-07** | Logistic 回归、线性回归 | P0 |
| **F-CS-08** | ANOVA/ANCOVA、t 检验、Wilcoxon、Kruskal-Wallis | P0 |
| **F-CS-09** | 倾向性评分匹配 | P2 |
| **F-CS-10** | 混合效应模型（纵向数据） | P1 |
| **F-CS-11** | Meta 分析（固定/随机效应、森林图、漏斗图） | P1 |
| **F-CS-12** | 样本量/功效计算 | P2 |
| **F-CS-13** | 多重检验校正（Bonferroni, FDR, 置换检验） | P0 |

### 3.8 多组学整合 (Multi-Omics Integration)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-MO-01** | MOFA/MOFA+ 多组学因子分析 | P0（Phase 4） |
| **F-MO-02** | DIABLO 监督式多组学判别分析 | P0（Phase 4） |
| **F-MO-03** | SNF 相似性网络融合 | P1 |
| **F-MO-04** | sPLS-DA / rCCA（mixOmics） | P1 |
| **F-MO-05** | iCluster/iClusterBayes 贝叶斯整合聚类 | P1 |
| **F-MO-06** | 多组学 ML（Random Forest / XGBoost + SHAP） | P1 |
| **F-MO-07** | WGCNA 多组学模块检测 | P1 |
| **F-MO-08** | Circos 图 / 高级多组学可视化 | P2 |

### 3.9 代理交互 (Agent Interaction)

| ID | 需求 | 优先级 |
|----|------|--------|
| **F-AI-01** | 自然语言交互式分析（CLI 对话） | P0 |
| **F-AI-02** | 分析方案确认（方法、参数、预期输出） | P0 |
| **F-AI-03** | 增量式分析（保持上下文，逐步深入） | P0 |
| **F-AI-04** | 分析结果解读（生物学/临床语境） | P0 |
| **F-AI-05** | 错误诊断与修复建议 | P0 |
| **F-AI-06** | 会话保存 / 恢复 / 导出 | P1 |
| **F-AI-07** | 脚本模式（YAML 工作流批量执行） | P1 |
| **F-AI-08** | 分析过程溯源与复现 | P1 |

### 3.10 项目感知交互模型 (Project-Aware Interaction Model)

PassiAgent 是一个面向生信项目目录的交互式分析代理。用户在存放生信数据集的文件夹下启动 CLI，Agent 通过「项目初始化 → 数据发现 → 交互规划 → 执行分析 → 结果解读 → 输出报告」六个阶段完成分析任务。

#### 3.10.1 交互流程概览

```
  ① 项目初始化          ② 数据发现             ③ 交互规划
  (CLI 启动)            (扫描工作目录)          (Agent 提议)
      ↓                     ↓                     ↓
  加载配置              检测文件格式            基于数据类型
  初始化 R 环境          识别组学领域            推荐分析方法
  创建会话              评估数据质量            用户确认/调整
      ↓                     ↓                     ↓
  ⑥ 输出报告            ⑤ 结果解读             ④ 执行分析
  (结果汇总)            (生物学解释)           (工具调用)
      ↓                     ↓                     ↓
  结构化输出目录         自然语言解读            ReAct 循环
  HTML/Markdown 报告     下游分析建议            增量式执行
  可复现追踪             临床意义评估            进度反馈
```

#### 3.10.2 阶段一：项目初始化 (Project Initialization)

```
# 用户在生信数据目录下启动
cd /project/covid19_rnaseq/
passi chat

# Agent 输出:
🧬 PassiAgent v0.1.0 — Multi-Omics Bioinformatics Analysis Agent
═══════════════════════════════════════════════════

[系统] 正在初始化项目环境...
[系统] R 4.6.0 环境就绪 — Bioconductor 3.23 已加载
[系统] 扫描工作目录: D:/project/covid19_rnaseq/

[系统] 发现 3 个数据文件:
  📊 GSE152075_raw_counts.txt.gz  (494 样本 × 58,289 基因) → 转录组 count matrix
  📋 metadata.csv                 (494 × 5)                   → 样本元数据
  📋 gene_annotations.gtf         (1.2 GB)                    → 基因注释

[系统] 检测到 1 个组学领域: 转录组学 (Transcriptomics)
[系统] 建议分析方向: 差异表达分析 → 富集分析 → 可视化
[系统] 会话已创建: 20260625_001
```

**F-AI-10**: 项目初始化 — Agent 启动时自动扫描工作目录，检测数据格式，识别组学领域 P0
**F-AI-11**: 环境状态报告 — 报告 R/Python 环境状态、可用包、计算资源 P1

#### 3.10.3 阶段二：数据发现与理解 (Data Discovery)

Agent 调用 `parse_omics_data` 对每个文件进行深度检测：

```
[Agent] 正在深入分析数据文件...

  counts.csv:
    • 格式: Count Matrix (genes × samples)
    • 维度: 58,289 基因 × 494 样本
    • 数据类型: integer (raw counts)
    • 缺失值: 0.2% (12,345 / 28,764,566)
    • 表达量范围: [0, 498,231]
    • 检测到 3,421 个低表达基因 (mean < 10 CPM)

  metadata.csv:
    • 列: sample_id, condition, timepoint, batch, age, sex
    • condition: COVID19 (n=245), Healthy (n=249)
    • timepoint: Day0 (n=98), Day3 (n=99), Day7 (n=99), Day14 (n=99), Day28 (n=99)
    • batch: Batch1–Batch5 (balanced)
```

**F-AI-12**: 数据深度检测 — 对每个数据文件进行格式、维度、类型、缺失值、分布特征分析 P0
**F-AI-13**: 数据质量评估 — 自动评估数据适用性，标注潜在问题（批次效应、缺失值、低质量样本）P1

#### 3.10.4 阶段三：交互式分析规划 (Interactive Planning)

Agent 基于数据发现结果，向用户提议分析方案并逐项确认：

```
[Agent] 基于您的转录组数据，我建议以下分析方案:

  📋 分析计划:
  1. 质量控制 (qc_report)
     - 检测批次效应 (PCA by batch)
     - 样本相关性热图

  2. 差异表达分析 (differential_analysis)
     - 方法: DESeq2 (推荐，适用于 raw counts)
     - 对比: COVID19 vs Healthy
     - 协变量: batch, age, sex
     - 阈值: FDR < 0.05, |log2FC| > 1

  3. 基因集富集分析 (enrichment_analysis)
     - GSEA 预排序分析 (fgsea)
     - GO + KEGG + Reactome

  4. 结果可视化
     - 火山图 (显著基因标注)
     - 热图 (Top 50 差异基因)
     - PCA 图 (分组着色)

  5. 输出报告

  请确认分析方案，或告诉我需要调整的地方。
  (回复 "ok" 或 "go" 开始执行，或指定修改项)

👤 > 第2步多加点参数：再加一个 timepoint 的对比，Day0 vs Day28

[Agent] 已更新分析方案:
  2a. 差异表达: COVID19 vs Healthy (主对比)
  2b. 差异表达: Day0 vs Day28 (时间点对比)
  确认开始执行? (ok / 继续调整)
```

**F-AI-14**: 分析方案提案 — 基于数据特征自动生成分析方案，包含方法推荐和参数建议 P0
**F-AI-15**: 交互式方案调整 — 用户可以增删改查分析步骤，Agent 实时更新方案 P0
**F-AI-16**: 方案确认机制 — 分析执行前需用户确认（或 `--auto-approve` 跳过）P0

#### 3.10.5 阶段四：分析执行 (Analysis Execution)

确认方案后，Agent 按步骤执行，每步显示进度和结果摘要：

```
[Agent] 开始执行分析方案...

━━━ Step 1/5: 质量控制 ━━━
🔧 qc_report(data_path="counts.csv", group_col="condition")

✓ QC 完成:
  • 缺失值: 0.2% (低风险)
  • PCA: PC1 解释 42% 方差，condition 是主要变异来源
  • 批次效应: batch 变量解释 8% 方差 (可接受)
  • 低表达基因: 3,421 个 (5.9%)，建议过滤
  • 建议: 应用 ComBat 批次校正

━━━ Step 2a/5: 差异表达 (COVID19 vs Healthy) ━━━
🔧 differential_analysis(method="deseq2", group_col="condition", covariates="batch, age, sex")
⏳ DESeq2 正在估计离散度...

✓ 差异分析完成 (耗时 47.2s):
  • 总基因: 54,868 (过滤低表达后)
  • 显著上调: 1,247 基因 (FDR < 0.05, log2FC > 1)
  • 显著下调: 892 基因
  • Top 基因: IFI27 (log2FC=5.2, padj=1.2e-42), ...
  • 结果文件: output/de_results_deseq2_counts.tsv

━━━ Step 3/5: 富集分析 ━━━
...
```

**F-AI-17**: 步骤式执行 — 按分析方案顺序执行，每步独立报告结果 P0
**F-AI-18**: 执行进度反馈 — 显示当前步骤、工具调用、耗时、关键指标 P1
**F-AI-19**: 中间结果可干预 — 用户可以在任意步骤后暂停、跳过、重做 P1

#### 3.10.6 阶段五：结果解读 (Result Interpretation)

每个分析步骤完成后，Agent 提供生物学/临床语境下的结果解读：

```
[Agent] 差异分析结果解读:

  🔬 生物学发现:
  • COVID-19 患者上调的 1,247 个基因主要富集在:
    - 干扰素信号通路 (IFN-α/β/γ, ISG 家族)
    - 炎症反应 (IL6, TNF, NF-κB)
    - 病毒防御 (OAS, MX, IFIT 家族)
  • 下调的 892 个基因主要包括:
    - T 细胞受体信号 (CD3, CD28)
    - 代谢通路 (OXPHOS, 脂肪酸氧化)

  📊 统计意义:
  • 效应量分布合理 (median |log2FC| = 1.8)
  • 多重检验校正后 2,139 基因显著 (FDR 5%)
  • 建议: 进一步分析 IFI27 作为潜在诊断标志物

  🔮 下游建议:
  • 对差异基因做 WGCNA 共表达网络分析
  • 与公开 COVID-19 数据集做 Meta-analysis
  • 考虑单细胞分辨率验证
```

**F-AI-20**: 生物学解读 — 结果以生物学/临床语境解读，而非纯统计报告 P0
**F-AI-21**: 下游分析建议 — 基于当前结果，建议后续分析方向 P1
**F-AI-22**: 文献参考 — (Phase 2+) 引用相关文献支持解读 P2

#### 3.10.7 阶段六：输出组织 (Output Organization)

分析完成后，Agent 将所有结果组织到结构化输出目录：

```
output/
└── 20260625_001_covid19_transcriptomics/     # session_id + project
    ├── analysis_report.html                   # 完整分析报告
    ├── analysis_report.md                     # Markdown 版本
    ├── session_summary.json                   # 会话摘要
    │
    ├── qc/                                    # QC 结果
    │   ├── qc_report.json
    │   ├── pca_plot.png
    │   └── correlation_heatmap.png
    │
    ├── differential_analysis/                 # 差异分析
    │   ├── de_results_deseq2_counts.tsv
    │   ├── volcano_plot.png
    │   └── heatmap_top50.png
    │
    ├── enrichment/                            # 富集分析
    │   ├── fgsea_results.tsv
    │   ├── fgsea_summary.json
    │   └── enrichment_barplot.png
    │
    ├── figures/                               # 汇总图表
    │   ├── figure_01_volcano.png
    │   ├── figure_02_heatmap.png
    │   └── figure_03_enrichment.png
    │
    └── provenance/                            # 溯源记录
        ├── wire.jsonl
        └── checksums.json
```

**F-AI-23**: 结构化输出 — 按分析步骤组织输出目录，每个步骤独立子目录 P0
**F-AI-24**: 分析报告 — 自动生成 HTML/Markdown 分析报告（含图表、方法、结果、解读）P0
**F-AI-25**: 溯源文件 — 记录所有分析步骤的 provenance（工具、参数、checksum）P1
**F-AI-26**: 中间结果保留 — 每个分析步骤的中间文件持久化，支持断点续跑 P1

#### 3.10.8 完整交互序列示例

```
                    ┌─────────────────────────────────────────┐
  👤 用户             │  🧬 Agent                              │
                    └─────────────────────────────────────────┘

  启动 CLI ──────────→ [1] 项目初始化: 扫描目录, 加载环境
                      [2] 数据发现: 解析文件, 评估质量
                      [3] 呈现项目摘要: "发现 3 个数据文件..."

  👤 "做差异分析" ──→ [4] 规划: 推荐 DESeq2, 请求确认分组列
                      "请指定分组列: condition 还是 timepoint?"

  👤 "condition" ────→ [5] 执行: 调用 differential_analysis 工具
                      ⏳ 进度: estimating dispersions...
                      [6] 展示结果: "1,247 上调, 892 下调"
                      [7] 解读: 生物学意义 + 下游建议

  👤 "做 GSEA" ──────→ [8] 执行: 调用 enrichment_analysis 工具
                      [9] 展示: "25 通路显著富集"
                      [10] 解读 + 建议

  👤 "/export" ──────→ [11] 输出: 生成报告 + 图表
                      "报告已保存到 output/20260625_001/"

  👤 "/quit" ────────→ [12] 清理: 保存会话, 记录 provenance
```



## 4. 非功能需求

### 4.1 性能

| ID | 需求 | 指标 |
|----|------|------|
| **NF-PERF-01** | CLI 冷启动时间 | < 1s |
| **NF-PERF-02** | LLM 首次响应时间 | < 3s（流式首字） |
| **NF-PERF-03** | Python/R 代码执行超时 | 可配置（默认 120s / 300s） |
| **NF-PERF-04** | 上下文窗口管理 | 自动压缩（>80K tokens 警告，>160K 关键） |
| **NF-PERF-05** | 大文件处理 | 支持流式读取，单次预览 ≤1000 行 |

### 4.2 可靠性

| ID | 需求 | 指标 |
|----|------|------|
| **NF-REL-01** | 工具参数校验 | Pydantic schema 强制校验，目标 >95% 参数有效性 |
| **NF-REL-02** | R 执行降级 | rpy2 不可用时自动切换 Rscript 子进程 |
| **NF-REL-03** | LLM 多供应商 | 主供应商不可用时降级到备用供应商 |
| **NF-REL-04** | 会话持久化 | 每 5 条消息自动 checkpoint |
| **NF-REL-05** | 分析溯源 | 每步记录 provenance（工具、参数、文件 checksum） |

### 4.3 可扩展性

| ID | 需求 | 指标 |
|----|------|------|
| **NF-EXT-01** | 工具即插即用 | 新工具注册：1 个文件 + 1 行注册 |
| **NF-EXT-02** | 方法目录可扩展 | 新方法添加到 `knowledge/methods.py` 即可被 LLM 发现 |
| **NF-EXT-03** | 新组学领域支持 | 添加工具文件 + 知识条目即可 |
| **NF-EXT-04** | 多 LLM 供应商 | 实现 LLMClient 抽象接口即可添加新供应商 |

### 4.4 安全性

| ID | 需求 | 指标 |
|----|------|------|
| **NF-SEC-01** | 代码执行沙箱 | 子进程隔离；Docker 可选 |
| **NF-SEC-02** | API Key 管理 | 环境变量 / .env 文件，不写死 |
| **NF-SEC-03** | 文件访问控制 | 仅限工作目录和用户指定路径 |
| **NF-SEC-04** | 输入长度限制 | 消息最大 100K 字符 |

### 4.5 可用性

| ID | 需求 | 指标 |
|----|------|------|
| **NF-UX-01** | 中文/英文双语支持 | 系统提示和错误信息双语 |
| **NF-UX-02** | 命令行补全 | 文件路径、命令名 Tab 补全 |
| **NF-UX-03** | 进度显示 | 长耗时操作（差异分析、MOFA）显示进度 |
| **NF-UX-04** | 颜色编码 | 不同消息角色（用户/代理/工具/错误）有区分色 |

---

## 5. 接口规格

### 5.1 CLI 接口

```
# 交互式对话模式（默认）
passi chat                          # 启动 Rich TUI 对话

# 非交互模式（stdout）
passi ask "对 counts.csv 做差异分析"  # 单次提问

# 批量工作流模式
passi run pipelines/rnaseq_diff_expr.yaml

# 会话管理
passi session list                  # 列出所有会话
passi session load <id>             # 恢复会话
passi session delete <id>           # 删除会话

# 工具直接调用
passi tool parse_omics_data --path counts.csv

# 知识库查询
passi knowledge search --query "差异表达"
passi knowledge methods --domain transcriptomics
passi knowledge formats --domain genomics
```

### 5.2 Python SDK（预留）

```python
from passi import PassiCLIent

client = PassiCLIent(provider="anthropic")
session = client.create_session(domain="multi-omics")

# 对话式分析
result = session.chat("对 counts.csv 做差异分析，group 在 metadata.csv")

# 直接调用工具
result = session.tool("parse_omics_data", path="counts.csv")

# 导出结果
session.export("analysis_report.html")
```

### 5.3 REST API（预留）

| 方法 | 路径 | 请求体 | 响应 |
|------|------|--------|------|
| POST | `/api/v1/sessions` | `{"domain": "transcriptomics"}` | `{"session_id": "..."}` |
| POST | `/api/v1/sessions/{id}/chat` | `{"message": "..."}` | `{"content": [...]}` |
| GET | `/api/v1/sessions/{id}` | — | `{"session_id": "...", "status": "..."}` |
| DELETE | `/api/v1/sessions/{id}` | — | `{"deleted": true}` |
| POST | `/api/v1/tools/{name}` | `{"params": {...}}` | `{"success": true, "result": {...}}` |
| GET | `/api/v1/knowledge/search?q=deseq2` | — | `{"methods": [...]}` |
| WS | `/api/v1/ws/{session_id}` | — | 实时双向通信 |

---

## 6. 数据流模型

### 6.1 典型分析流程

```
用户上传数据 → 格式自动检测 → 数据预览 → QC检查
    ↓
分析方案确认（方法、参数）
    ↓
单组学分析（差异/富集/网络/生存...）
    ↓
结果解读 → 可视化 → 导出
    ↓
（可选）多组学整合 → 生物标志物发现 → 临床解读
```

### 6.2 会话状态机

```
[创建会话] → [活跃] ←→ [暂停（checkpoint）]
                ↓
            [完成] / [归档]
```

---

## 7. 约束与假设

### 7.1 技术约束

- Python 3.11+（类型注解、asyncio 改进）
- R 4.2+ 环境（用于 Bioconductor 方法）
- 需要网络连接（LLM API 调用）
- 部分工具需要特定软件（PLINK, Docker）

### 7.2 业务假设

- 用户具备基本的组学数据格式知识
- 数据文件为常见生物信息学格式（如第 2 节所列）
- 用户负责数据的前期预处理（比对、定量已完成）
- Agent 不替代专业生物信息学家，而是辅助加速常规分析

---

## 8. 软件开发过程：V 模型 (V-Model)

本项目采用 V 模型（V-Model）进行软件开发，每个开发阶段对应一个测试阶段，确保需求→设计→实现→验证的可追溯性。

### 8.1 V 模型结构

```
  需求分析                  验收测试
  (Requirements)          (Acceptance Test)
       ↓                        ↑
   系统设计                  系统测试
  (System Design)         (System Test)
       ↓                        ↑
   架构设计                  集成测试
  (Architecture)         (Integration Test)
       ↓                        ↑
   模块设计                  单元测试
  (Module Design)         (Unit Test)
       ↓                        ↑
   编码实现 ──── TDD ────→
       (Coding)
```

### 8.2 各阶段对应关系

| V 模型阶段 | 开发活动 | 输出物 | 对应测试阶段 | 测试活动 | 测试依据 |
|-----------|---------|--------|-------------|---------|---------|
| **需求分析** | 用户场景定义、功能需求规格 | `specification.md` 第3节 功能需求 | **验收测试 (UAT)** | 端到端场景验证 | 验收标准 (Section 9) |
| **系统设计** | 接口规格、数据流、非功能需求 | `specification.md` 第4-5节 | **系统测试 (ST)** | API测试、性能测试、安全测试 | 非功能需求指标 |
| **架构设计** | 分层架构、组件定义、通信模式 | `architecture.md` | **集成测试 (IT)** | 组件间通信、数据流、Wire协议 | 架构规格 |
| **模块设计** | 类/函数接口、Pydantic模型、工具契约 | `design.md` + 源码接口 | **单元测试 (UT)** | 函数级测试、工具单元测试、模型校验 | 模块接口定义 |
| **编码实现** | TDD 红-绿-重构循环 | 源码 + 测试代码 | — | 同步进行 | 模块设计 |

### 8.3 V 模型阶段详细说明

#### 左翼：开发阶段 (Decomposition)

**Phase 1 — 需求分析 (Requirements Analysis)**
- 输入：用户需求、领域分析
- 活动：功能需求分解（F-DATA-*, F-TX-*, F-CS-* 等）、验收标准定义
- 输出：`specification.md` — 功能需求 + 验收标准
- 出口条件：所有功能需求已明确 ID、优先级、验收条件

**Phase 2 — 系统设计 (System Design)**
- 输入：功能需求规格
- 活动：CLI/API/SDK 接口设计、数据流模型、非功能需求量化
- 输出：`specification.md` 第4-5节 — 接口规格 + 非功能需求指标
- 出口条件：所有接口已定义请求/响应格式、性能指标已量化

**Phase 3 — 架构设计 (Architecture Design)**
- 输入：系统设计规格
- 活动：分层架构定义、组件职责划分、通信协议设计、技术选型
- 输出：`architecture.md` — 架构图 + 组件说明 + 数据流
- 出口条件：所有组件间接口已定义、Wire 事件类型已枚举

**Phase 4 — 模块设计 (Module Design)**
- 输入：架构设计规格
- 活动：类/函数接口定义、Pydantic 模型设计、工具契约制定、测试用例设计
- 输出：`design.md` + 源码接口定义 + 测试用例
- 出口条件：每个模块的公开 API 已定义、参数模型已通过 schema 校验

**Phase 5 — 编码实现 (Coding with TDD)**
- 输入：模块设计规格
- 活动：TDD 红-绿-重构循环
- 输出：源码 + 单元测试 + 集成测试
- 出口条件：所有测试通过、覆盖率 ≥ 80%

#### 右翼：测试阶段 (Integration)

**Phase 5' — 单元测试 (Unit Testing)**
- 对应模块设计阶段
- 范围：单个函数/类/工具
- 工具：pytest, pytest-asyncio, pytest-cov
- 目标：每个公开方法 ≥ 1 个 happy path + 1 个 error case

**Phase 4' — 集成测试 (Integration Testing)**
- 对应架构设计阶段
- 范围：组件间交互（Agent↔ToolRegistry, Wire↔Persistence, RExecutor↔rpy2）
- 工具：pytest + mock fixtures
- 目标：每个组件间通信路径被测试

**Phase 3' — 系统测试 (System Testing)**
- 对应系统设计阶段
- 范围：完整的 CLI/API 功能链路
- 工具：pytest + subprocess
- 目标：所有 CLI 命令可正常执行、API 端点返回正确状态码

**Phase 2' — 验收测试 (Acceptance Testing)**
- 对应需求分析阶段
- 范围：端到端用户场景
- 依据：Section 9 验收标准
- 目标：每个验收场景通过（"RNA-seq差异分析"、"生存分析"等）

### 8.4 V 模型项目里程碑

| 里程碑 | V 模型阶段 | 关键交付物 | 检查点 |
|--------|-----------|-----------|--------|
| M1 | 需求分析完成 | `specification.md` 定稿 | 所有功能需求 ID 化 |
| M2 | 系统设计完成 | 接口规格 + 非功能需求 | 所有接口请求/响应格式确定 |
| M3 | 架构设计完成 | `architecture.md` 定稿 | 组件间通信协议确定 |
| M4 | 模块设计完成 | `design.md` + 测试用例 | 每个模块的公开 API 确定 |
| M5 | Alpha 版本 | 核心功能实现 + 单元测试 | 单元测试覆盖率 ≥ 80% |
| M6 | Beta 版本 | 集成测试通过 + 系统测试通过 | 端到端场景可运行 |
| M7 | RC 版本 | 验收测试通过 | 所有验收标准满足 |
| M8 | V1.0 发布 | 文档 + 测试 + 部署 | 所有 V 模型阶段完成 |

---

## 9. 验收标准

| 场景 | 验收条件 |
|------|---------|
| RNA-seq 差异分析 | 输入 count matrix + 分组信息 → 输出差异基因列表 + 火山图 |
| GSEA 富集分析 | 输入差异基因列表 → 输出富集通路 + 条形图 |
| 生存分析 | 输入临床数据 + 分组 → KM 曲线 + log-rank p 值 + Cox HR |
| 多组学 MOFA | 输入 2-3 个组学矩阵 → 输出潜在因子 + 方差分解 |
| 格式检测 | 拖入任意支持格式 → 正确识别组学类型和格式 |
| 会话恢复 | 关闭 → 重新打开 → 恢复之前的分析上下文 |
