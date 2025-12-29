const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

// Color scheme - Professional Academic Blue
const colors = {
    primary: '1C2833',      // Deep navy
    secondary: '2E4053',    // Slate gray
    accent: '16A085',       // Teal
    highlight: 'E74C3C',    // Red accent
    lightBg: 'F8F9FA',      // Light gray
    white: 'FFFFFF',
    text: '2C3E50',         // Dark text
    subtext: '7F8C8D'       // Gray text
};

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.title = '基于自然梯度提升的社交媒体评论热度预测';
    pptx.author = '作者';

    // ========== Slide 1: 封面 ==========
    let slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: '100%', fill: { color: colors.primary } });
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 4.2, w: '100%', h: 1.4, fill: { color: colors.accent } });
    slide.addText('基于自然梯度提升的\n社交媒体评论热度预测', {
        x: 0.5, y: 1.2, w: 9, h: 2,
        fontSize: 40, bold: true, color: colors.white,
        align: 'center', valign: 'middle', fontFace: 'Arial'
    });
    slide.addText('以小米SU7微博数据为例', {
        x: 0.5, y: 3.2, w: 9, h: 0.6,
        fontSize: 24, color: colors.lightBg,
        align: 'center', fontFace: 'Arial'
    });
    slide.addText('2024年', {
        x: 0.5, y: 4.5, w: 9, h: 0.5,
        fontSize: 18, color: colors.white,
        align: 'center', fontFace: 'Arial'
    });

    // ========== Slide 2: 研究背景 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('研究背景', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0.3, y: 1.3, w: 0.08, h: 3.8, fill: { color: colors.accent } });

    slide.addText([
        { text: '社交媒体热度预测的重要性\n', options: { bold: true, fontSize: 20, color: colors.text } },
        { text: '• 舆情监控与品牌管理\n', options: { fontSize: 16, color: colors.text } },
        { text: '• 内容推荐系统优化\n', options: { fontSize: 16, color: colors.text } },
        { text: '• 信息传播规律研究\n\n', options: { fontSize: 16, color: colors.text } },
        { text: '小米SU7发布引发热议\n', options: { bold: true, fontSize: 20, color: colors.text } },
        { text: '• 2024年3月28日正式发布\n', options: { fontSize: 16, color: colors.text } },
        { text: '• 小米首款新能源汽车\n', options: { fontSize: 16, color: colors.text } },
        { text: '• 社交媒体广泛讨论\n\n', options: { fontSize: 16, color: colors.text } },
        { text: '数据特点\n', options: { bold: true, fontSize: 20, color: colors.text } },
        { text: '• 长尾分布：少数热门评论获得大量互动\n', options: { fontSize: 16, color: colors.text } },
        { text: '• 时效性强：热度随时间快速衰减', options: { fontSize: 16, color: colors.text } }
    ], { x: 0.6, y: 1.4, w: 8.8, h: 4, valign: 'top', fontFace: 'Arial' });

    // ========== Slide 3: 研究挑战 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('研究挑战', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Three challenge boxes
    const boxY = 1.5;
    const boxH = 3.2;
    const boxW = 2.9;
    const gap = 0.2;

    // Box 1
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: boxY, w: boxW, h: boxH, fill: { color: 'ECF0F1' }, line: { color: colors.accent, pt: 2 }, rectRadius: 0.1 });
    slide.addText('长尾分布问题', { x: 0.5, y: boxY + 0.2, w: boxW, h: 0.5, fontSize: 16, bold: true, color: colors.accent, align: 'center', fontFace: 'Arial' });
    slide.addText('• 传统MSE不适用\n• 大数预测微小误差被过度惩罚\n• 需要关注相对误差', { x: 0.6, y: boxY + 0.8, w: boxW - 0.2, h: 2, fontSize: 13, color: colors.text, fontFace: 'Arial' });

    // Box 2
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5 + boxW + gap, y: boxY, w: boxW, h: boxH, fill: { color: 'ECF0F1' }, line: { color: colors.highlight, pt: 2 }, rectRadius: 0.1 });
    slide.addText('不确定性量化', { x: 0.5 + boxW + gap, y: boxY + 0.2, w: boxW, h: 0.5, fontSize: 16, bold: true, color: colors.highlight, align: 'center', fontFace: 'Arial' });
    slide.addText('• 模型需要知道"自己不知道什么"\n• 输出预测置信度\n• 概率预测而非点估计', { x: 0.6 + boxW + gap, y: boxY + 0.8, w: boxW - 0.2, h: 2, fontSize: 13, color: colors.text, fontFace: 'Arial' });

    // Box 3
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5 + 2*(boxW + gap), y: boxY, w: boxW, h: boxH, fill: { color: 'ECF0F1' }, line: { color: colors.secondary, pt: 2 }, rectRadius: 0.1 });
    slide.addText('特征表示', { x: 0.5 + 2*(boxW + gap), y: boxY + 0.2, w: boxW, h: 0.5, fontSize: 16, bold: true, color: colors.secondary, align: 'center', fontFace: 'Arial' });
    slide.addText('• 多模态特征融合\n• 文本、用户、时序信息\n• 重复/相似内容检测', { x: 0.6 + 2*(boxW + gap), y: boxY + 0.8, w: boxW - 0.2, h: 2, fontSize: 13, color: colors.text, fontFace: 'Arial' });

    // ========== Slide 4: 数据集构建 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('数据集构建', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Left column - data source
    slide.addText('数据来源', { x: 0.5, y: 1.3, w: 4.3, h: 0.4, fontSize: 18, bold: true, color: colors.accent, fontFace: 'Arial' });
    slide.addText('• 新浪微博平台\n• 时间范围：2024.3.27 - 2024.4.14\n• 采集方式：隧道代理绕过反爬虫\n• 检查点机制支持断点续传', { x: 0.5, y: 1.8, w: 4.3, h: 1.5, fontSize: 14, color: colors.text, fontFace: 'Arial' });

    // Right column - statistics
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 5, y: 1.3, w: 4.3, h: 2, fill: { color: colors.accent }, rectRadius: 0.1 });
    slide.addText('271,452', { x: 5, y: 1.5, w: 4.3, h: 0.7, fontSize: 36, bold: true, color: colors.white, align: 'center', fontFace: 'Arial' });
    slide.addText('条评论数据', { x: 5, y: 2.2, w: 4.3, h: 0.4, fontSize: 18, color: colors.white, align: 'center', fontFace: 'Arial' });

    // Table for data split
    slide.addText('数据集划分 (8:1:1)', { x: 0.5, y: 3.5, w: 9, h: 0.4, fontSize: 18, bold: true, color: colors.text, fontFace: 'Arial' });

    const tableData = [
        [{ text: '数据集', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: '样本数', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: '占比', options: { fill: { color: colors.primary }, color: colors.white, bold: true } }],
        ['训练集', '217,162', '80%'],
        ['验证集', '27,145', '10%'],
        ['测试集', '27,145', '10%']
    ];
    slide.addTable(tableData, { x: 0.5, y: 4.0, w: 8.5, h: 1.2, colW: [3, 3, 2.5], border: { pt: 1, color: 'CCCCCC' }, fontSize: 14, align: 'center', valign: 'middle', fontFace: 'Arial' });

    // ========== Slide 5: 特征工程 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('特征工程（4类17维）', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Feature boxes - 2x2 grid
    const fBoxW = 4.4;
    const fBoxH = 1.8;
    const fBoxGap = 0.2;

    // Box 1 - 基础特征
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 1.2, w: fBoxW, h: fBoxH, fill: { color: 'E8F6F3' }, line: { color: colors.accent, pt: 2 }, rectRadius: 0.08 });
    slide.addText('基础特征 (7维)', { x: 0.5, y: 1.3, w: fBoxW - 0.2, h: 0.35, fontSize: 15, bold: true, color: colors.accent, fontFace: 'Arial' });
    slide.addText('用户评论数、是否认证、是否一级评论\n微博评论数、发布时间（小时/星期/工作日）', { x: 0.5, y: 1.7, w: fBoxW - 0.2, h: 1.1, fontSize: 12, color: colors.text, fontFace: 'Arial' });

    // Box 2 - 文本特征
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4 + fBoxW + fBoxGap, y: 1.2, w: fBoxW, h: fBoxH, fill: { color: 'FDF2E9' }, line: { color: 'E67E22', pt: 2 }, rectRadius: 0.08 });
    slide.addText('文本特征 (6维)', { x: 0.5 + fBoxW + fBoxGap, y: 1.3, w: fBoxW - 0.2, h: 0.35, fontSize: 15, bold: true, color: 'E67E22', fontFace: 'Arial' });
    slide.addText('评论长度、感叹号数、问号数\n表情数、话题标签、小米关键词数', { x: 0.5 + fBoxW + fBoxGap, y: 1.7, w: fBoxW - 0.2, h: 1.1, fontSize: 12, color: colors.text, fontFace: 'Arial' });

    // Box 3 - LDA主题
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 1.2 + fBoxH + fBoxGap, w: fBoxW, h: fBoxH, fill: { color: 'F5EEF8' }, line: { color: '9B59B6', pt: 2 }, rectRadius: 0.08 });
    slide.addText('LDA主题 (1维)', { x: 0.5, y: 1.3 + fBoxH + fBoxGap, w: fBoxW - 0.2, h: 0.35, fontSize: 15, bold: true, color: '9B59B6', fontFace: 'Arial' });
    slide.addText('潜在狄利克雷分配主题模型\n性能讨论 / 安全话题 / 品牌对比...', { x: 0.5, y: 1.7 + fBoxH + fBoxGap, w: fBoxW - 0.2, h: 1.1, fontSize: 12, color: colors.text, fontFace: 'Arial' });

    // Box 4 - 时间密度
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4 + fBoxW + fBoxGap, y: 1.2 + fBoxH + fBoxGap, w: fBoxW, h: fBoxH, fill: { color: 'EBF5FB' }, line: { color: '3498DB', pt: 2 }, rectRadius: 0.08 });
    slide.addText('时间密度 (3维)', { x: 0.5 + fBoxW + fBoxGap, y: 1.3 + fBoxH + fBoxGap, w: fBoxW - 0.2, h: 0.35, fontSize: 15, bold: true, color: '3498DB', fontFace: 'Arial' });
    slide.addText('时间顺序索引、最大相似度、重复次数\nMinHash算法高效检测相似评论', { x: 0.5 + fBoxW + fBoxGap, y: 1.7 + fBoxH + fBoxGap, w: fBoxW - 0.2, h: 1.1, fontSize: 12, color: colors.text, fontFace: 'Arial' });

    // ========== Slide 6: MinHash相似度检测 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('MinHash相似度检测', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    slide.addText([
        { text: '算法原理\n', options: { bold: true, fontSize: 18, color: colors.accent } },
        { text: '• 文本转换为N-gram集合 (N=3)\n', options: { fontSize: 14, color: colors.text } },
        { text: '• 使用128个哈希函数计算MinHash签名\n', options: { fontSize: 14, color: colors.text } },
        { text: '• 通过Jaccard相似度估计文本相似度\n\n', options: { fontSize: 14, color: colors.text } },
        { text: '优化策略\n', options: { bold: true, fontSize: 18, color: colors.accent } },
        { text: '• 滑动窗口：维护近10,000条评论的签名\n', options: { fontSize: 14, color: colors.text } },
        { text: '• 全局TopK：记录高频重复文本（出现>3次）\n', options: { fontSize: 14, color: colors.text } },
        { text: '• "老梗"记忆机制\n', options: { fontSize: 14, color: colors.text } }
    ], { x: 0.5, y: 1.3, w: 5, h: 3.5, valign: 'top', fontFace: 'Arial' });

    // Complexity comparison box
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 5.5, y: 1.5, w: 4, h: 2.5, fill: { color: 'E8F8F5' }, line: { color: colors.accent, pt: 2 }, rectRadius: 0.1 });
    slide.addText('复杂度对比', { x: 5.5, y: 1.6, w: 4, h: 0.4, fontSize: 16, bold: true, color: colors.accent, align: 'center', fontFace: 'Arial' });
    slide.addText('暴力计算\nO(n²)', { x: 5.7, y: 2.1, w: 1.7, h: 0.8, fontSize: 14, color: colors.highlight, align: 'center', fontFace: 'Arial' });
    slide.addText('→', { x: 7.3, y: 2.3, w: 0.4, h: 0.4, fontSize: 24, color: colors.text, align: 'center', fontFace: 'Arial' });
    slide.addText('MinHash\nO(n·k)', { x: 7.7, y: 2.1, w: 1.7, h: 0.8, fontSize: 14, color: colors.accent, align: 'center', fontFace: 'Arial' });
    slide.addText('k = 哈希函数数量 (128)', { x: 5.5, y: 3.4, w: 4, h: 0.4, fontSize: 12, color: colors.subtext, align: 'center', fontFace: 'Arial' });

    // ========== Slide 7: NGBoost模型 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('NGBoost模型', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    slide.addText([
        { text: '自然梯度提升 (Natural Gradient Boosting)\n', options: { bold: true, fontSize: 18, color: colors.accent } },
        { text: '• 基于梯度提升的概率预测方法\n', options: { fontSize: 14, color: colors.text } },
        { text: '• 直接拟合条件概率分布参数\n', options: { fontSize: 14, color: colors.text } },
        { text: '• 同时输出预测均值 μ 和标准差 σ\n\n', options: { fontSize: 14, color: colors.text } }
    ], { x: 0.5, y: 1.2, w: 9, h: 1.5, valign: 'top', fontFace: 'Arial' });

    // Loss function box
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 2.6, w: 9, h: 1.3, fill: { color: 'FEF9E7' }, line: { color: 'F1C40F', pt: 2 }, rectRadius: 0.1 });
    slide.addText('对数尺度NLL损失函数', { x: 0.6, y: 2.7, w: 8.8, h: 0.35, fontSize: 15, bold: true, color: 'B7950B', fontFace: 'Arial' });
    slide.addText('L = 0.5 · log(σ²) + [log(y+10) - log(μ+10)]² / (2σ²)', { x: 0.6, y: 3.1, w: 8.8, h: 0.5, fontSize: 18, color: colors.text, align: 'center', fontFace: 'Courier New' });

    // Config table
    slide.addText('模型配置', { x: 0.5, y: 4.1, w: 9, h: 0.35, fontSize: 16, bold: true, color: colors.text, fontFace: 'Arial' });
    const configTable = [
        [{ text: '参数', options: { fill: { color: colors.secondary }, color: colors.white, bold: true } },
         { text: '值', options: { fill: { color: colors.secondary }, color: colors.white, bold: true } }],
        ['n_estimators', '100'],
        ['max_depth', '10'],
        ['learning_rate', '0.1']
    ];
    slide.addTable(configTable, { x: 0.5, y: 4.5, w: 4, h: 0.9, colW: [2, 2], border: { pt: 1, color: 'CCCCCC' }, fontSize: 12, align: 'center', valign: 'middle', fontFace: 'Arial' });

    // ========== Slide 8: BGE神经网络架构 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('BGE神经网络架构', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Architecture flow
    slide.addText('模型架构流程', { x: 0.5, y: 1.2, w: 9, h: 0.35, fontSize: 16, bold: true, color: colors.text, fontFace: 'Arial' });

    // Flow boxes
    const flowY = 1.7;
    const flowBoxW = 1.5;
    const flowBoxH = 0.7;
    const arrowW = 0.3;

    const flowItems = ['4类文本', '预处理', 'BGE编码', 'Cross-Att', '特征拼接', '双预测头', '(μ, σ)'];
    const flowColors = [colors.secondary, '3498DB', colors.accent, '9B59B6', 'E67E22', colors.highlight, colors.primary];

    for (let i = 0; i < flowItems.length; i++) {
        const x = 0.3 + i * (flowBoxW + arrowW);
        slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: x, y: flowY, w: flowBoxW, h: flowBoxH, fill: { color: flowColors[i] }, rectRadius: 0.08 });
        slide.addText(flowItems[i], { x: x, y: flowY, w: flowBoxW, h: flowBoxH, fontSize: 11, bold: true, color: colors.white, align: 'center', valign: 'middle', fontFace: 'Arial' });
        if (i < flowItems.length - 1) {
            slide.addText('→', { x: x + flowBoxW, y: flowY, w: arrowW, h: flowBoxH, fontSize: 18, color: colors.text, align: 'center', valign: 'middle', fontFace: 'Arial' });
        }
    }

    // Details
    slide.addText([
        { text: 'BGE-base-zh-v1.5\n', options: { bold: true, fontSize: 16, color: colors.accent } },
        { text: '• 768维中文文本嵌入\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 编码4类文本：评论/微博/根评论/父评论\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 默认冻结，支持微调\n', options: { fontSize: 13, color: colors.text } }
    ], { x: 0.5, y: 2.6, w: 4.3, h: 1.4, valign: 'top', fontFace: 'Arial' });

    slide.addText([
        { text: 'VIP用户白名单\n', options: { bold: true, fontSize: 16, color: colors.highlight } },
        { text: '• 19个高频用户保留原始ID\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 雷军、小米官方、卢伟冰...\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 其他@用户替换为 _USER_\n', options: { fontSize: 13, color: colors.text } }
    ], { x: 5, y: 2.6, w: 4.3, h: 1.4, valign: 'top', fontFace: 'Arial' });

    slide.addText([
        { text: 'Cross-Attention融合\n', options: { bold: true, fontSize: 16, color: '9B59B6' } },
        { text: '• 评论向量作为Query\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 上下文（微博/父评论）作为Key/Value\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 自适应融合上下文信息\n', options: { fontSize: 13, color: colors.text } }
    ], { x: 0.5, y: 4.1, w: 4.3, h: 1.2, valign: 'top', fontFace: 'Arial' });

    slide.addText([
        { text: '双预测头\n', options: { bold: true, fontSize: 16, color: 'E67E22' } },
        { text: '• 均值头：预测子评论数\n', options: { fontSize: 13, color: colors.text } },
        { text: '• 方差头：预测不确定性\n', options: { fontSize: 13, color: colors.text } },
        { text: '• Softplus激活保证正值\n', options: { fontSize: 13, color: colors.text } }
    ], { x: 5, y: 4.1, w: 4.3, h: 1.2, valign: 'top', fontFace: 'Arial' });

    // ========== Slide 9: 评价指标体系 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('评价指标体系', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Two columns
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 1.3, w: 4.4, h: 3.8, fill: { color: 'E8F8F5' }, line: { color: colors.accent, pt: 2 }, rectRadius: 0.1 });
    slide.addText('精度指标', { x: 0.4, y: 1.4, w: 4.4, h: 0.4, fontSize: 18, bold: true, color: colors.accent, align: 'center', fontFace: 'Arial' });
    slide.addText([
        { text: 'MSLE (均方对数误差)\n', options: { bold: true, fontSize: 14, color: colors.text } },
        { text: '关注相对误差，适合长尾分布\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: 'ACP@20% (容忍区间准确率)\n', options: { bold: true, fontSize: 14, color: colors.text } },
        { text: '|y - ŷ| ≤ max(20%·y, 5)\n', options: { fontSize: 12, color: colors.subtext, fontFace: 'Courier New' } },
        { text: '直观反映预测实用价值', options: { fontSize: 12, color: colors.subtext } }
    ], { x: 0.5, y: 1.9, w: 4.2, h: 3, valign: 'top', fontFace: 'Arial' });

    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.3, w: 4.4, h: 3.8, fill: { color: 'FDF2E9' }, line: { color: 'E67E22', pt: 2 }, rectRadius: 0.1 });
    slide.addText('不确定性指标', { x: 5.2, y: 1.4, w: 4.4, h: 0.4, fontSize: 18, bold: true, color: 'E67E22', align: 'center', fontFace: 'Arial' });
    slide.addText([
        { text: 'NLL (负对数似然)\n', options: { bold: true, fontSize: 14, color: colors.text } },
        { text: '评估真实值在预测分布中的概率\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: 'PICP@95% (置信区间覆盖率)\n', options: { bold: true, fontSize: 14, color: colors.text } },
        { text: '真实值落在95%置信区间的比例\n', options: { fontSize: 12, color: colors.subtext } },
        { text: '理想值接近95%', options: { fontSize: 12, color: colors.subtext } }
    ], { x: 5.3, y: 1.9, w: 4.2, h: 3, valign: 'top', fontFace: 'Arial' });

    // ========== Slide 10: 实验结果 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('实验结果', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Results table
    const resultsTable = [
        [{ text: '数据集', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: 'R²', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: 'MSLE', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: 'ACP@20%', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: 'PICP@95%', options: { fill: { color: colors.primary }, color: colors.white, bold: true } }],
        ['训练集', '0.652', '0.0179', '98.15%', '96.95%'],
        [{ text: '验证集', options: { bold: true } }, { text: '0.191', options: { bold: true } }, { text: '0.0283', options: { bold: true } }, { text: '97.61%', options: { bold: true, color: colors.accent } }, { text: '94.14%', options: { bold: true, color: colors.accent } }],
        [{ text: '测试集', options: { bold: true } }, { text: '0.015', options: { bold: true } }, { text: '0.0347', options: { bold: true } }, { text: '97.61%', options: { bold: true, color: colors.accent } }, { text: '94.44%', options: { bold: true, color: colors.accent } }]
    ];
    slide.addTable(resultsTable, { x: 0.5, y: 1.3, w: 9, h: 1.6, colW: [1.8, 1.8, 1.8, 1.8, 1.8], border: { pt: 1, color: 'CCCCCC' }, fontSize: 14, align: 'center', valign: 'middle', fontFace: 'Arial' });

    // Key findings
    slide.addText('关键发现', { x: 0.5, y: 3.2, w: 9, h: 0.4, fontSize: 18, bold: true, color: colors.text, fontFace: 'Arial' });
    slide.addText([
        { text: '• ACP@20% 达到 97.61%：', options: { fontSize: 14, color: colors.text } },
        { text: '绝大多数预测在20%容忍范围内\n', options: { fontSize: 14, color: colors.accent } },
        { text: '• PICP@95% 接近 95%：', options: { fontSize: 14, color: colors.text } },
        { text: '不确定性估计校准良好\n', options: { fontSize: 14, color: colors.accent } },
        { text: '• R² 较低但符合预期：', options: { fontSize: 14, color: colors.text } },
        { text: '社交媒体数据固有的随机性\n', options: { fontSize: 14, color: colors.subtext } },
        { text: '• 训练集与验证集差距：', options: { fontSize: 14, color: colors.text } },
        { text: '提示可能存在过拟合，但泛化仍可接受', options: { fontSize: 14, color: colors.subtext } }
    ], { x: 0.5, y: 3.6, w: 9, h: 1.8, valign: 'top', fontFace: 'Arial' });

    // ========== Slide 11: 特征重要性 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('特征重要性分析', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Feature importance chart data
    const featureData = [{
        name: '重要性',
        labels: ['用户评论数', '是否一级评论', '微博评论数', '评论长度', '时间索引', '发布小时', '最大相似度', '表情数', '感叹号数', '重复次数'],
        values: [0.314, 0.266, 0.211, 0.058, 0.042, 0.035, 0.028, 0.018, 0.015, 0.013]
    }];
    slide.addChart(pptx.charts.BAR, featureData, {
        x: 0.5, y: 1.2, w: 5.5, h: 3.8,
        barDir: 'bar',
        showTitle: false,
        showLegend: false,
        showCatAxisTitle: false,
        showValAxisTitle: true,
        valAxisTitle: '重要性得分',
        valAxisMaxVal: 0.35,
        chartColors: [colors.accent],
        dataLabelPosition: 'outEnd',
        dataLabelFontSize: 10,
        catAxisLabelFontSize: 11
    });

    // Insights
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 6.2, y: 1.3, w: 3.4, h: 3.5, fill: { color: 'F8F9FA' }, line: { color: colors.secondary, pt: 1 }, rectRadius: 0.1 });
    slide.addText('关键洞察', { x: 6.2, y: 1.4, w: 3.4, h: 0.4, fontSize: 16, bold: true, color: colors.secondary, align: 'center', fontFace: 'Arial' });
    slide.addText([
        { text: 'Top 3 特征占比 79%\n\n', options: { bold: true, fontSize: 13, color: colors.accent } },
        { text: '用户活跃度\n', options: { bold: true, fontSize: 12, color: colors.text } },
        { text: '活跃用户的评论更易获关注\n\n', options: { fontSize: 11, color: colors.subtext } },
        { text: '评论层级\n', options: { bold: true, fontSize: 12, color: colors.text } },
        { text: '一级评论曝光度更高\n\n', options: { fontSize: 11, color: colors.subtext } },
        { text: '微博热度\n', options: { bold: true, fontSize: 12, color: colors.text } },
        { text: '热门微博下的评论互动更多', options: { fontSize: 11, color: colors.subtext } }
    ], { x: 6.3, y: 1.9, w: 3.2, h: 2.8, valign: 'top', fontFace: 'Arial' });

    // ========== Slide 12: 基线对比 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('基线方法对比', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Comparison table
    const baselineTable = [
        [{ text: '方法', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: 'R²', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: 'MSLE', options: { fill: { color: colors.primary }, color: colors.white, bold: true } },
         { text: '不确定性', options: { fill: { color: colors.primary }, color: colors.white, bold: true } }],
        ['Ridge', '0.082', '0.0312', '×'],
        ['Lasso', '0.079', '0.0315', '×'],
        ['Random Forest', '0.156', '0.0295', '×'],
        ['GBDT', '0.168', '0.0290', '×'],
        ['XGBoost', '0.175', '0.0288', '×'],
        ['LightGBM', '0.178', '0.0286', '×'],
        [{ text: 'NGBoost', options: { bold: true, color: colors.accent } },
         { text: '0.191', options: { bold: true, color: colors.accent } },
         { text: '0.0283', options: { bold: true, color: colors.accent } },
         { text: '✓', options: { bold: true, color: colors.accent } }]
    ];
    slide.addTable(baselineTable, { x: 0.5, y: 1.3, w: 9, h: 2.8, colW: [2.5, 2, 2.5, 2], border: { pt: 1, color: 'CCCCCC' }, fontSize: 14, align: 'center', valign: 'middle', fontFace: 'Arial' });

    // Highlight box
    slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.3, w: 9, h: 0.9, fill: { color: 'E8F8F5' }, line: { color: colors.accent, pt: 2 }, rectRadius: 0.1 });
    slide.addText('NGBoost 不仅在预测精度上优于所有基线方法，还是唯一支持不确定性估计的方法', { x: 0.6, y: 4.4, w: 8.8, h: 0.7, fontSize: 15, color: colors.text, align: 'center', valign: 'middle', fontFace: 'Arial' });

    // ========== Slide 13: 结论与展望 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 1.0, fill: { color: colors.primary } });
    slide.addText('结论与展望', { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, bold: true, color: colors.white, fontFace: 'Arial' });

    // Conclusions
    slide.addText('主要贡献', { x: 0.5, y: 1.2, w: 4.3, h: 0.4, fontSize: 18, bold: true, color: colors.accent, fontFace: 'Arial' });
    slide.addText([
        { text: '1. 数据集构建\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '   27万条小米SU7微博评论\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: '2. 特征工程\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '   4类17维互补特征\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: '3. 概率预测\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '   NGBoost + 对数NLL损失\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: '4. 评价体系\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '   多维指标全面评估', options: { fontSize: 12, color: colors.subtext } }
    ], { x: 0.5, y: 1.7, w: 4.3, h: 3.2, valign: 'top', fontFace: 'Arial' });

    // Future work
    slide.addText('未来工作', { x: 5.2, y: 1.2, w: 4.3, h: 0.4, fontSize: 18, bold: true, color: 'E67E22', fontFace: 'Arial' });
    slide.addText([
        { text: '用户画像\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '社交网络结构、历史互动模式\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: '时序建模\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '捕捉热度动态演化规律\n\n', options: { fontSize: 12, color: colors.subtext } },
        { text: '跨平台验证\n', options: { bold: true, fontSize: 13, color: colors.text } },
        { text: '推广到其他社交媒体平台', options: { fontSize: 12, color: colors.subtext } }
    ], { x: 5.2, y: 1.7, w: 4.3, h: 3.2, valign: 'top', fontFace: 'Arial' });

    // ========== Slide 14: 致谢 ==========
    slide = pptx.addSlide();
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: '100%', fill: { color: colors.primary } });
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 3.8, w: '100%', h: 1.8, fill: { color: colors.accent } });
    slide.addText('感谢聆听', { x: 0, y: 1.8, w: 10, h: 1, fontSize: 48, bold: true, color: colors.white, align: 'center', fontFace: 'Arial' });
    slide.addText('欢迎提问与讨论', { x: 0, y: 4.2, w: 10, h: 0.6, fontSize: 24, color: colors.white, align: 'center', fontFace: 'Arial' });

    // Save
    const outputPath = path.join(__dirname, '..', 'presentation.pptx');
    await pptx.writeFile({ fileName: outputPath });
    console.log('Presentation created successfully at:', outputPath);
}

createPresentation().catch(console.error);
