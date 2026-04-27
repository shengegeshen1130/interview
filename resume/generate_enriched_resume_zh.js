const { Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, BorderStyle } = require("docx");
const fs = require("fs");

const LINE = 228;
const BODY = 16;
const BSPC = { before: 2, after: 2, line: LINE };

const numbering = {
  config: [
    {
      reference: "bullets",
      levels: [
        {
          level: 0,
          format: LevelFormat.BULLET,
          text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: {
            paragraph: {
              indent: { left: 300, hanging: 170 },
            },
          },
        },
      ],
    },
  ],
};

function bullet(text) {
  const children = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) {
      children.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Microsoft YaHei", size: BODY }));
    } else {
      children.push(new TextRun({ text: part, font: "Microsoft YaHei", size: BODY }));
    }
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: BSPC,
    children,
  });
}

function sectionHeader(text) {
  return new Paragraph({
    spacing: { before: 70, after: 24 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 1, color: "333333", space: 1 },
    },
    children: [
      new TextRun({ text, bold: true, font: "Microsoft YaHei", size: 20, color: "1A1A1A" }),
    ],
  });
}

function projectTitle(title, dates) {
  return new Paragraph({
    spacing: { before: 36, after: 8, line: LINE },
    children: [
      new TextRun({ text: title, bold: true, font: "Microsoft YaHei", size: 17 }),
      new TextRun({ text: `  (${dates})`, italics: true, font: "Microsoft YaHei", size: BODY, color: "555555" }),
    ],
  });
}

function r(text, opts = {}) {
  return new TextRun({ text, font: "Microsoft YaHei", size: opts.size || BODY, ...opts });
}

const doc = new Document({
  numbering,
  styles: {
    default: {
      document: {
        run: { font: "Microsoft YaHei", size: 18 },
      },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 720, bottom: 580, left: 960, right: 960 },
        },
      },
      children: [
        // ── 姓名 ──
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 0, line: 256 },
          children: [
            new TextRun({ text: "\u845B  \u71CA", bold: true, font: "Microsoft YaHei", size: 32, color: "1A1A1A" }),
          ],
        }),

        // ── 联系方式 ──
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 16, line: LINE },
          children: [
            r("shengegeshen@gmail.com  |  +86-186-0205-4952  |  \u4E0A\u6D77 \u2192 \u65B0\u52A0\u5761 (2025\u5E749\u6708)", { color: "444444" }),
          ],
        }),

        // ── 个人简介 ──
        sectionHeader("\u4E2A\u4EBA\u7B80\u4ECB"),
        new Paragraph({
          spacing: { before: 8, after: 30, line: LINE },
          children: [
            r("\u9AD8\u7EA7 ML/AI \u5DE5\u7A0B\u5E08\uFF0C\u62E5\u67097\u5E74\u4EE5\u4E0A PayPal \u751F\u4EA7\u7EA7\u673A\u5668\u5B66\u4E60\u53CA\u5927\u8BED\u8A00\u6A21\u578B\u7CFB\u7EDF\u7684\u8BBE\u8BA1\u4E0E\u843D\u5730\u7ECF\u9A8C\uFF0C\u4E1A\u52A1\u8986\u76D6\u6B3A\u8BC8\u68C0\u6D4B\u3001\u53CD\u6D17\u94B1\u53CA\u91D1\u878D\u98CE\u63A7\u9886\u57DF\u3002\u6DF1\u5165\u638C\u63E1\u591A\u667A\u80FD\u4F53 LLM \u7CFB\u7EDF\u7F16\u6392\uFF08LangChain\u3001LangGraph\uFF09\u3001\u56FE\u795E\u7ECF\u7F51\u7EDC\uFF08GNN\u3001\u793E\u533A\u53D1\u73B0\uFF09\u4EE5\u53CA\u6A21\u578B\u5168\u751F\u547D\u5468\u671F\u7BA1\u7406\u3002\u73B0\u5BFB\u6C42 AI \u5DE5\u7A0B\u5E08\u5C97\u4F4D\uFF0C\u5E0C\u671B\u5C06\u751F\u4EA7\u7EA7 LLM \u4E0E\u673A\u5668\u5B66\u4E60\u5DE5\u7A0B\u7ECF\u9A8C\u5E94\u7528\u4E8E\u4E0B\u4E00\u4EE3\u667A\u80FD\u7CFB\u7EDF\u3002"),
          ],
        }),

        // ── 技术技能 ──
        sectionHeader("\u6280\u672F\u6280\u80FD"),
        new Paragraph({
          spacing: { before: 8, after: 0, line: LINE },
          children: [
            r("LLM \u4E0E\u667A\u80FD\u4F53: ", { bold: true }),
            r("LangChain, LangGraph, DSPy, Prompt Engineering, RAG, Graph-RAG, Multi-Agent, Chain-of-Thought"),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 0, line: LINE },
          children: [
            r("\u673A\u5668\u5B66\u4E60\u4E0E\u6DF1\u5EA6\u5B66\u4E60: ", { bold: true }),
            r("PyTorch, DGL, GNN (GAT/GCN), \u6A21\u578B\u84B8\u998F, AutoEncoder, LSTM, LightGBM, XGBoost, PU-Learning, \u5F02\u5E38\u68C0\u6D4B, SHAP"),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 0, line: LINE },
          children: [
            r("\u6570\u636E\u4E0E\u57FA\u7840\u8BBE\u65BD: ", { bold: true }),
            r("BigQuery, Spark, Neo4j, DGL, Gremlin, Airflow, Docker, GCP, \u6A21\u578B\u670D\u52A1\u5316"),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 30, line: LINE },
          children: [
            r("\u7F16\u7A0B\u8BED\u8A00\u4E0E\u5DE5\u5177: ", { bold: true }),
            r("Python, Scala, Java, SQL, Git, Jupyter, PandasAI, Scikit-learn"),
          ],
        }),

        // ── 工作经历 ──
        sectionHeader("\u5DE5\u4F5C\u7ECF\u5386"),
        new Paragraph({
          spacing: { before: 24, after: 0, line: LINE },
          children: [
            new TextRun({ text: "PayPal", bold: true, font: "Microsoft YaHei", size: 19 }),
            r("  \u2014  \u9AD8\u7EA7\u673A\u5668\u5B66\u4E60\u5DE5\u7A0B\u5E08", { size: 18 }),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 8, line: LINE },
          children: [
            r("\u4E0A\u6D77 (2019.02 \u2013 2025.08)  |  \u65B0\u52A0\u5761 (2025.09 \u2013 \u81F3\u4ECA)", { italics: true, color: "555555" }),
          ],
        }),

        // ── AML 智能调查系统 ──
        projectTitle("AML \u667A\u80FD\u8C03\u67E5\u7CFB\u7EDF \u2014 \u591A\u667A\u80FD\u4F53\u5927\u8BED\u8A00\u6A21\u578B\u7CFB\u7EDF", "2025.09 \u2013 \u81F3\u4ECA"),
        bullet("\u57FA\u4E8E LangChain Deep Agent \u8BBE\u8BA1\u5E76\u642D\u5EFA**\u591A\u667A\u80FD\u4F53\u67B6\u6784**\uFF08\u4E3B\u667A\u80FD\u4F53\u8D1F\u8D23\u4EFB\u52A1\u89C4\u5212\u4E0E\u6267\u884C\u8C03\u5EA6\uFF0C\u642D\u914D React \u98CE\u683C\u7684\u4E13\u4E1A\u5316\u5B50\u667A\u80FD\u4F53\uFF09\uFF0C\u5B9E\u73B0\u53CD\u6D17\u94B1\u6848\u4EF6\u7684\u7AEF\u5230\u7AEF\u81EA\u52A8\u5316\u8C03\u67E5\u6D41\u7A0B\u2014\u2014\u6DB5\u76D6\u8BC1\u636E\u68C0\u7D22\u3001\u6CD5\u89C4\u89C4\u5219\u63A8\u7406\u3001\u98CE\u9669\u7B5B\u67E5\u4EE5\u53CA\u8C03\u67E5\u62A5\u544A\u751F\u6210\u2014\u2014**\u51B3\u7B56\u51C6\u786E\u7387\u8FBE 80%**\uFF0C\u4E0E\u8D44\u6DF1\u8C03\u67E5\u5458\u7684\u5224\u65AD\u6C34\u5E73\u6301\u5E73\u3002"),
        bullet("\u6784\u5EFA **Graph-RAG \u77E5\u8BC6\u5C42**\uFF0C\u5C06 50 \u4EFD\u4EE5\u4E0A\u5185\u90E8\u64CD\u4F5C\u89C4\u7A0B\u4EE5\u53CA\u5916\u90E8\u53CD\u6D17\u94B1\u6CD5\u5F8B\u6CD5\u89C4\u7F16\u7801\u4E3A\u7ED3\u6784\u5316\u77E5\u8BC6\u56FE\u8C31\uFF08\u89E6\u53D1\u6761\u4EF6 \u2192 \u5904\u7406\u64CD\u4F5C \u2192 \u6CD5\u5F8B\u6761\u6587\uFF09\uFF0C\u4F7F\u667A\u80FD\u4F53\u5177\u5907\u57FA\u4E8E\u5408\u89C4\u8981\u6C42\u7684\u63A8\u7406\u80FD\u529B\u3002"),
        bullet("\u501F\u52A9 DSPy \u5B9E\u73B0\u63D0\u793A\u8BCD\u7684\u81EA\u52A8\u5316\u4F18\u5316\uFF0C\u5E76\u642D\u5EFA\u8FED\u4EE3\u8BC4\u4F30\u6D41\u7A0B\u6301\u7EED\u6539\u8FDB\u667A\u80FD\u4F53\u8868\u73B0\uFF0C\u6700\u7EC8\u5C06\u5355\u6848\u5BA1\u67E5\u65F6\u95F4\u4ECE **3\u20134 \u5C0F\u65F6\u7F29\u77ED\u81F3 30 \u5206\u949F\u4EE5\u5185**\uFF08\u6548\u7387\u63D0\u5347\u7EA6 85%\uFF09\uFF0C\u8BE5\u7ED3\u679C\u7ECF 15 \u540D\u8C03\u67E5\u5458\u5B9E\u9645\u4F7F\u7528\u9A8C\u8BC1\u3002"),
        bullet("\u652F\u6301\u8C03\u67E5\u5458\u4E0E\u667A\u80FD\u4F53\u8FDB\u884C**\u591A\u8F6E\u4EA4\u4E92\u5F0F\u5BF9\u8BDD**\uFF0C\u53EF\u6DF1\u5165\u67E5\u8BE2\u6848\u4EF6\u7EC6\u8282\u3001\u8C03\u6574\u5206\u6790\u53C2\u6570\u3001\u9010\u6B65\u4FEE\u6539\u548C\u5B8C\u5584\u8C03\u67E5\u62A5\u544A\u8349\u7A3F\uFF0C\u6574\u4E2A\u8FC7\u7A0B\u4FDD\u7559\u5B8C\u6574\u7684\u5BA1\u8BA1\u8F68\u8FF9\u3002"),

        // ── 自然语言转 SQL ──
        projectTitle("\u81EA\u7136\u8BED\u8A00\u8F6C SQL \u8C03\u67E5\u5DE5\u5177 (Buyer Risk) \u2014 LLM \u9A71\u52A8", "2025.01 \u2013 2025.08"),
        bullet("\u8BBE\u8BA1**\u591A\u6B65\u9AA4\u68C0\u7D22\u589E\u5F3A\u751F\u6210\u6D41\u6C34\u7EBF**\uFF08\u5148\u5BF9\u7528\u6237\u67E5\u8BE2\u8FDB\u884C\u8BED\u4E49\u6539\u5199\uFF0C\u518D\u901A\u8FC7 BGE-M3 \u5411\u91CF\u68C0\u7D22\u76F8\u4F3C\u67E5\u8BE2\u6837\u4F8B\uFF0C\u6700\u540E\u7ED3\u5408\u601D\u7EF4\u94FE\u63D0\u793A\u751F\u6210 BigQuery SQL\uFF09\uFF0C**\u751F\u6210\u51C6\u786E\u7387\u8FBE 94%**\uFF0C\u8C03\u67E5\u5458\u4EBA\u5DE5\u7F16\u5199 SQL \u7684\u65F6\u95F4\u51CF\u5C11\u4E86 **80%**\u3002"),
        bullet("\u57FA\u4E8E LangGraph \u6784\u5EFA**\u6709\u72B6\u6001\u7684\u591A\u667A\u80FD\u4F53\u5DE5\u4F5C\u6D41**\uFF0C\u5C06\u610F\u56FE\u8BC6\u522B\u4E0E\u8DEF\u7531\u3001\u9519\u8BEF\u81EA\u52A8\u4FEE\u590D\uFF883 \u8F6E Dry Run \u53CD\u9988\u5FAA\u73AF\uFF09\u3001\u4EA4\u4E92\u5F0F SQL \u4FEE\u6539\u5B50\u56FE\u3001PandasAI \u6570\u636E\u53EF\u89C6\u5316\u5206\u6790\u4EE5\u53CA\u5386\u53F2\u53CD\u9988\u590D\u7528\u7B49\u529F\u80FD\u6574\u5408\u4E3A\u4E00\u4E2A\u7EDF\u4E00\u7684\u4EBA\u673A\u534F\u540C\u8C03\u67E5\u5E73\u53F0\u3002"),

        // ── 主动欺诈检测 ──
        projectTitle("\u4E3B\u52A8\u6B3A\u8BC8\u68C0\u6D4B \u2014 \u57FA\u4E8E\u56FE\u7684\u8D26\u6237\u5173\u8054\u4E0E\u805A\u7C7B", "2023.06 \u2013 2024.12"),
        bullet("\u4E3B\u5BFC\u5F00\u53D1\u57FA\u4E8E\u56FE\u7684\u6B3A\u8BC8\u68C0\u6D4B\u6D41\u6C34\u7EBF\uFF0C\u5904\u7406\u89C4\u6A21\u8FBE **1 \u4EBF\u4EE5\u4E0A\u8D26\u6237\u30013 \u4EBF\u6761\u5173\u8054\u8FB9**\uFF0C\u5148\u540E\u7ECF\u5386 4 \u4EE3\u805A\u7C7B\u65B9\u6CD5\u7684\u6F14\u8FDB\uFF08\u79CD\u5B50\u8D26\u6237\u5339\u914D \u2192 Gremlin \u591A\u8DF3\u56FE\u904D\u5386 \u2192 Louvain \u793E\u533A\u53D1\u73B0\u7B97\u6CD5 \u2192 \u8D26\u6237\u5D4C\u5165\u5411\u91CF\u76F8\u4F3C\u5EA6\u8BA1\u7B97\uFF09\uFF0C\u5728\u8D26\u6237\u751F\u547D\u5468\u671F\u65E9\u671F\u4E3B\u52A8\u8BC6\u522B\u6B3A\u8BC8\u56E2\u4F19\u2014\u2014\u5E74\u5316**\u51C0\u635F\u5931\u8282\u7701\u8D85\u8FC7 $20M**\u3002"),
        bullet("\u901A\u8FC7\u6A21\u578B\u84B8\u998F\u6280\u672F\u8BAD\u7EC3 AutoEncoder \u63D0\u53D6\u8D26\u6237\u7EA7\u522B\u7684\u5D4C\u5165\u5411\u91CF\u8868\u793A\uFF0C\u7136\u540E\u5728\u5173\u8054\u56FE\u4E0A\u5E94\u7528**\u8FB9\u611F\u77E5\u56FE\u6CE8\u610F\u529B\u7F51\u7EDC GAT**\uFF08DGL/PyTorch\uFF09\u2014\u2014\u5C06\u8FB9\u7279\u5F81\uFF08\u5173\u8054\u7C7B\u578B\u3001\u8D44\u4EA7\u98CE\u9669\u5EA6\uFF09\u76F4\u63A5\u62FC\u63A5\u81F3\u6CE8\u610F\u529B\u5C42\u8F93\u51FA\u4EE5\u5B8C\u6574\u4FDD\u7559\u8FB9\u4FE1\u606F\u2014\u2014\u540C\u65F6\u8BA1\u7B97\u7EC4\u7EA7\u548C\u8D26\u6237\u7EA7\u98CE\u9669\u5206\uFF1B\u5E76\u5C06\u8C03\u67E5\u5458\u8D44\u6E90\u5206\u914D\u95EE\u9898\u5EFA\u6A21\u4E3A**\u6574\u6570\u7EBF\u6027\u89C4\u5212**\u4F18\u5316\u6C42\u89E3\u3002"),

        // ── ML 模型开发 ──
        projectTitle("\u591A\u98CE\u63A7\u9886\u57DF ML \u6A21\u578B\u5F00\u53D1 \u2014 \u5168\u573A\u666F\u98CE\u9669\u5EFA\u6A21", "2019.02 \u2013 2023.05"),
        bullet("\u72EC\u7ACB\u8D1F\u8D23**\u591A\u4E2A\u98CE\u63A7\u4E1A\u52A1\u573A\u666F**\u7684\u6A21\u578B\u5168\u751F\u547D\u5468\u671F\u7BA1\u7406\uFF08\u76D7\u5237\u68C0\u6D4B\u3001\u8D26\u6237\u63A5\u7BA1\u3001\u4E70\u5BB6\u8FDD\u89C4\u884C\u4E3A\u3001\u4E32\u8C0B\u6B3A\u8BC8\u3001\u5546\u6237\u7F51\u7AD9\u5408\u89C4\u5BA1\u67E5\u3001\u4EA4\u6613\u4E89\u8BAE\u81EA\u52A8\u5316\u7B49\uFF09\uFF0C\u6839\u636E\u6BCF\u4E2A\u9886\u57DF\u7684\u4E1A\u52A1\u7279\u70B9\u548C\u6570\u636E\u7279\u5F81\u7075\u6D3B\u9009\u62E9\u548C\u8C03\u6574\u5EFA\u6A21\u7B56\u7565\u2014\u2014\u4ECE\u6807\u7B7E\u4F53\u7CFB\u8BBE\u8BA1\u3001\u7279\u5F81\u5DE5\u7A0B\u5230\u6A21\u578B\u8BAD\u7EC3\u3001\u8BC4\u4F30\u4E0E\u751F\u4EA7\u90E8\u7F72\u3002"),
        bullet("\u6839\u636E\u4E0D\u540C\u4E1A\u52A1\u9700\u6C42\u7075\u6D3B\u8FD0\u7528**\u591A\u79CD\u5EFA\u6A21\u65B9\u6CD5**\uFF1A\u901A\u8FC7 PU-Learning \u89E3\u51B3\u6B63\u6837\u672C\u6807\u7B7E\u4E0D\u5B8C\u6574\u7684\u95EE\u9898\uFF1B\u8BBE\u8BA1\u795E\u7ECF\u7F51\u7EDC\u7684\u635F\u5931\u51FD\u6570\u52A0\u6743\u4E0E\u6837\u672C\u6743\u91CD\u5E73\u8861\u7B56\u7565\u4F18\u5148\u5B66\u4E60\u9AD8\u635F\u5931\u6B3A\u8BC8\u6837\u672C\uFF1B\u4F7F\u7528 LSTM \u7F16\u89E3\u7801\u5668\u5BF9\u7528\u6237\u884C\u4E3A\u5E8F\u5217\u8FDB\u884C\u5F02\u5E38\u68C0\u6D4B\uFF1B\u501F\u52A9 Word2Vec \u751F\u6210\u4E70\u5BB6\u884C\u4E3A\u5D4C\u5165\u5E76\u7ED3\u5408 KMeans \u8FDB\u884C\u884C\u4E3A\u805A\u7C7B\uFF1B\u4EE5 LightGBM/XGBoost \u4F5C\u4E3A\u5FEB\u901F\u57FA\u7EBF\u6A21\u578B\u5E76\u901A\u8FC7\u795E\u7ECF\u7F51\u7EDC\u8FED\u4EE3\u4F18\u5316\u3002"),
        bullet("\u6784\u5EFA**\u5546\u6237\u7F51\u7AD9\u98CE\u63A7\u8BC4\u5206\u7CFB\u7EDF**\uFF0C\u57FA\u4E8E\u722C\u53D6\u7684\u975E\u7ED3\u6784\u5316 HTML \u9875\u9762\u5185\u5BB9\u548C\u5916\u90E8\u6D41\u91CF\u6570\u636E\u8FDB\u884C\u98CE\u9669\u8BC4\u4F30\uFF1B\u8BBE\u8BA1\u5B9A\u5236\u5316\u7279\u5F81\u5DE5\u7A0B\u6D41\u6C34\u7EBF\uFF08\u5361 BIN \u98CE\u9669\u8BC4\u5206\u3001\u8BBE\u5907\u6307\u7EB9\u8BC6\u522B\u3001IP \u5730\u7406\u4F4D\u7F6E\u5F02\u5E38\u68C0\u6D4B\u3001\u4EA4\u6613\u5907\u6CE8 BERT \u5D4C\u5165\uFF09\uFF1B\u4EA4\u4ED8\u57FA\u4E8E SHAP \u7684\u53EF\u89E3\u91CA AI \u8F93\u51FA\uFF0C\u6EE1\u8DB3\u5185\u90E8\u5BA1\u67E5\u548C\u5408\u89C4\u8981\u6C42\u3002"),

        // ── 团队管理 ──
        new Paragraph({
          spacing: { before: 24, after: 8, line: LINE },
          children: [
            r("\u56E2\u961F\u7BA1\u7406: ", { bold: true }),
            r("\u6307\u5BFC 2 \u540D\u521D\u7EA7\u6570\u636E\u79D1\u5B66\u5BB6\uFF0C\u5E2E\u52A9\u5176\u5FEB\u901F\u638C\u63E1 ML \u6A21\u578B\u5F00\u53D1\u6D41\u7A0B\u548C\u98CE\u63A7\u4E1A\u52A1\u77E5\u8BC6\u3002\u7EC4\u7EC7\u5E76\u4E3B\u5BFC PayPal \u63A8\u8350\u7CFB\u7EDF\u5B66\u4E60\u5C0F\u7EC4\uFF082024\uFF09\uFF0C\u63A8\u52A8\u8DE8\u56E2\u961F\u7684\u6280\u672F\u5206\u4EAB\u4E0E\u77E5\u8BC6\u4EA4\u6D41\u3002"),
          ],
        }),

        // ── 教育背景 ──
        sectionHeader("\u6559\u80B2\u80CC\u666F"),
        new Paragraph({
          spacing: { before: 24, after: 0, line: LINE },
          children: [
            r("\u9999\u6E2F\u4E2D\u6587\u5927\u5B66", { bold: true, size: 17 }),
            r("  \u2014  \u5546\u4E1A\u5206\u6790\u7855\u58EB  |  2017 \u2013 2018  |  GPA: 3.40/4.0"),
          ],
        }),
        new Paragraph({
          spacing: { before: 8, after: 0, line: LINE },
          children: [
            r("\u5A01\u5EC9\u4E0E\u739B\u4E3D\u5B66\u9662 (College of William & Mary)", { bold: true, size: 17 }),
            r("  \u2014  \u5E94\u7528\u6570\u5B66\u5B66\u58EB\uFF08\u8BA1\u7B97\u673A\u8F85\u4FEE\uFF09  |  2013 \u2013 2017  |  GPA: 3.62/4.0\uFF0C6\u00d7 Dean's List"),
          ],
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = "/Users/shenge/gits/interview/resume/ShenGE_resume_enriched_zh.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Chinese enriched resume saved to: " + outPath);
});
