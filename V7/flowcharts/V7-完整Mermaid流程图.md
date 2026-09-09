# V7 完整 Mermaid 流程图

统一使用黑白灰样式，以分区标题说明继承部分和 V7 新增策略。实线表示主流程，虚线表示补充的数据或标签连接。

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#ffffff", "primaryBorderColor": "#888888", "primaryTextColor": "#222222", "lineColor": "#777777", "clusterBkg": "#fafafa", "clusterBorder": "#aaaaaa", "edgeLabelBackground": "#ffffff"}, "flowchart": {"htmlLabels": true, "curve": "linear"}}}%%
flowchart TB
    DATA["BNCI2014001 左右手二分类<br/>1296 × 22 × 1001；250 Hz"]
    LOOP["5 个随机种子 × 9 折 LOSO<br/>每折重新初始化模型及优化器"]
    DATA --> LOOP
    LOOP --> SRC["8 名源受试者<br/>1152 条 EEG＋源标签＋受试者 ID"]
    LOOP --> TAR["1 名留出目标受试者<br/>144 条 EEG；标签仅用于评估"]

    subgraph PREP["① 源训练数据准备：继承 V5"]
        SRC --> EA["逐源受试者 EA 对齐"]
        EA --> STYLE["每条样本：22 通道 × 4 时间窗<br/>提取 log-RMS 风格"]
        STYLE --> PROTO["构建源受试者 × 类别风格原型<br/>分组均值的中位数；8 × 2 × 22 × 4"]
        PROTO --> AUG["以概率 0.47 扰动源样本<br/>混合其他源受试者的同类原型<br/>强度 0.60；缩放限制 0.80–1.25"]
        EA --> AUG
        AUG --> FIXED["插值为逐通道时间缩放曲线并乘原信号<br/>每折预先生成一次；标签不变"]
        FIXED --> BATCH["打乱取源训练 batch<br/>batch_size = 32"]
    end

    subgraph MODEL["② 三视图网络：保留 V0／V2／V5 的竞争 Softmax 拼接结构"]
        BATCH --> INPUT["模型输入 B × 1 × 22 × 1001<br/>去掉单通道维度 → B × 22 × 1001"]
        INPUT --> T0["时间嵌入＋位置编码<br/>8 个 token：B × 8 × 40"]
        INPUT --> S0["电极嵌入＋位置编码<br/>22 个 token：B × 22 × 40"]
        INPUT --> F0["5 个可学习带通滤波器<br/>初始化频带：4–8／8–13／13–20／20–30／30–40 Hz"]
        T0 --> T1["第 1 个时间 Transformer"]
        S0 --> S1["第 1 个空间 Transformer"]
        F0 --> F1["共享频带深度卷积<br/>log-power → 投影＋频带嵌入<br/>5 个频带 token：B × 5 × 40"]
        T1 --> ZT["时间均值池化 zT<br/>B × 40"]
        S1 --> ZS["空间注意力池化 zS<br/>B × 40"]
        F1 --> ZF["频带注意力池化 zF<br/>B × 40"]
        ZT --> STACK["堆叠 3 个视图摘要<br/>B × 3 × 40"]
        ZS --> STACK
        ZF --> STACK
        STACK --> INTER["三视图瓶颈交互<br/>4 头自注意力＋FFN＋残差／归一化"]
        INTER --> SPLIT["拆分交互摘要<br/>zT′、zS′、zF′"]
        SPLIT --> TF["时间残差反馈<br/>原 token＋gT × zT′"]
        SPLIT --> SF["空间残差反馈<br/>原 token＋gS × zS′"]
        SPLIT --> FF["频率摘要残差反馈<br/>FF = zF＋gF × zF′"]
        T1 -. 原 token .-> TF
        S1 -. 原 token .-> SF
        ZF -. 原摘要 .-> FF
        TF --> T2["第 2 个时间 Transformer<br/>均值池化 → FT：B × 40"]
        SF --> S2["第 2 个空间 Transformer<br/>注意力池化 → FS：B × 40"]
        T2 --> CAT["拼接 FT、FS、FF<br/>B × 120"]
        S2 --> CAT
        FF --> CAT
        CAT --> GATE["融合门控 MLP：120 → 16 → 3<br/>竞争 Softmax：wT＋wS＋wF = 1<br/>初始权重均为 1/3"]
        GATE --> FUSE["逐视图加权后拼接<br/>concat：wT·FT，wS·FS，wF·FF<br/>B × 120；不是加权求和"]
        T2 -. FT .-> FUSE
        S2 -. FS .-> FUSE
        FF -. FF .-> FUSE
        FUSE --> HEAD["分类头：120 → 64 → 32 → 2<br/>ELU＋Dropout"]
        HEAD --> LOGITS["源样本 logits：B × 2"]
    end

    subgraph TRAIN["③ 源域监督训练：V7 新增优化策略"]
        LABEL["对应 batch 的源标签"]
        LOGITS --> CE["交叉熵损失 CE<br/>无 V9 对齐损失"]
        LABEL --> CE
        CE --> GRAD["清零梯度 → 反向传播"]
        LR["按每次迭代设置学习率<br/>前 5 轮线性 warmup 至 0.001<br/>后 95 轮 cosine 衰减至 0"]
        GRAD --> UPDATE["AdamW 更新模型参数<br/>weight_decay = 0.0001"]
        LR --> UPDATE
        UPDATE --> STEP{"本轮 batch 全部完成？"}
        STEP -- 否：下一 batch --> BATCH
        STEP -- 是 --> EVAL["切换 eval 模式<br/>目标集前向计算；无梯度"]
    end

    subgraph TEST["④ 目标评估与保存：不将目标标签传入源训练"]
        TAR --> TEA["从原始目标 EEG 构建在线 EA<br/>按顺序累计至当前样本的协方差<br/>不做源风格扰动"]
        TEA --> EVAL
        EVAL --> PRED["使用同一三视图网络的当前参数<br/>目标 logits → argmax 预测"]
        YT["目标真实标签<br/>仅用于计算指标"]
        TAR -. 标签分离 .-> YT
        PRED --> METRIC["计算目标 Accuracy<br/>每轮记录 train loss／train ACC／test ACC／LR"]
        YT --> METRIC
        METRIC --> DONE{"已训练 100 轮？"}
        DONE -- 否：恢复 train 模式 --> BATCH
        DONE -- 是 --> SAVE["保存第 100 轮模型及最终 ACC<br/>保存 epoch CSV、曲线及日志<br/>不按目标 ACC 峰值选模型"]
        SAVE --> NEXT{"45 次运行全部完成？"}
        NEXT -- 否：下一受试者或种子 --> LOOP
        NEXT -- 是 --> RESULT["汇总 5 × 9 准确率<br/>逐受试者均值／逐种子均值<br/>V7：78.37963% ± 0.29035"]
    end

    BATCH -. 对应标签 .-> LABEL
    LEGEND["说明：模型结构与源扰动继承 V5<br/>V7 新增 AdamW、warmup 和 cosine 策略<br/>反馈门 gT／gS／gF 使用 sigmoid，初始为 0.05<br/>末端融合门使用 Softmax；V7 不含 V4 频率 token 细化"]
    RESULT --- LEGEND

    classDef base fill:#ffffff,stroke:#888888,color:#222222;
    classDef aug fill:#ffffff,stroke:#888888,color:#222222;
    classDef novel fill:#ffffff,stroke:#888888,color:#222222;
    classDef neutral fill:#ffffff,stroke:#888888,color:#222222;
    class INPUT,T0,S0,F0,T1,S1,F1,ZT,ZS,ZF,STACK,INTER,SPLIT,TF,SF,FF,T2,S2,CAT,GATE,FUSE,HEAD,LOGITS base;
    class EA,STYLE,PROTO,AUG,FIXED aug;
    class LR,UPDATE novel;
    class DATA,LOOP,SRC,TAR,BATCH,LABEL,CE,GRAD,STEP,EVAL,TEA,PRED,YT,METRIC,DONE,SAVE,NEXT,RESULT,LEGEND neutral;
```

## 阅读说明

- 这是单折训练的完整展开；外层为 5 个种子、9 折 LOSO，共 45 次独立模型训练。B 表示当前批次样本数。
- 目标评估节点复用图中同一三视图网络，不是新增另一个网络。在线 EA 使用当前及此前目标样本的信号统计，不使用目标标签。
- 源扰动在每折数据加载时生成一次，之后重复使用，不是每个 epoch 重新生成。
- 反馈门使用 sigmoid；最终三视图融合门使用竞争 Softmax。V7 没有增加频率 token 细化、源验证选轮或类别条件对齐损失。
- 当前代码每轮计算目标 ACC，但不以此反向传播、调学习率或选择检查点；最终报告第 100 轮结果。反复根据同一目标集结果挑选版本仍可能产生实验选择偏差。
- 结果中的标准差为五个种子各自九受试者平均 ACC 的总体标准差，不是 45 项 ACC 的标准差。

## 代码依据

- [训练入口与学习率策略](../DBConformer_LOSO.py)
- [三视图前向过程](../models/TriViewDBConformer.py)
- [Softmax 加权拼接](../models/TriViewConcatDBConformer.py)
- [源受试者风格扰动](../utils/vision_dg_preprocess.py)
- [EA、数据加载与目标在线评估数据](../utils/utils.py)

图依据本地代码静态核对；Mermaid 源码已检查节点和连接引用，未使用外部渲染器验证排版。
