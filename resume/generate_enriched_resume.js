const { Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, BorderStyle } = require("docx");
const fs = require("fs");

// Bullet numbering config
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
              indent: { left: 360, hanging: 200 },
            },
          },
        },
      ],
    },
  ],
};

function bullet(text, boldParts) {
  // Parse text for **bold** markers
  const children = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) {
      children.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Calibri", size: 18 }));
    } else {
      children.push(new TextRun({ text: part, font: "Calibri", size: 18 }));
    }
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 10, after: 10, line: 252 },
    children,
  });
}

function sectionHeader(text) {
  return new Paragraph({
    spacing: { before: 120, after: 40 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 1, color: "333333", space: 1 },
    },
    children: [
      new TextRun({ text, bold: true, font: "Calibri", size: 22, color: "1A1A1A" }),
    ],
  });
}

function projectTitle(title, dates) {
  return new Paragraph({
    spacing: { before: 60, after: 20, line: 252 },
    children: [
      new TextRun({ text: title, bold: true, font: "Calibri", size: 19 }),
      new TextRun({ text: `  (${dates})`, italics: true, font: "Calibri", size: 18, color: "555555" }),
    ],
  });
}

const doc = new Document({
  numbering,
  styles: {
    default: {
      document: {
        run: { font: "Calibri", size: 20 },
      },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 860, bottom: 720, left: 1080, right: 1080 },
        },
      },
      children: [
        // ── Name ──
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 0, line: 276 },
          children: [
            new TextRun({ text: "SHEN GE", bold: true, font: "Calibri", size: 36, color: "1A1A1A" }),
          ],
        }),

        // ── Contact ──
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 40, line: 252 },
          children: [
            new TextRun({ text: "1339105051@qq.com  |  +86-186-0205-4952  |  Shanghai \u2192 Singapore (Sep 2025)", font: "Calibri", size: 18, color: "444444" }),
          ],
        }),

        // ── Professional Summary ──
        sectionHeader("PROFESSIONAL SUMMARY"),
        new Paragraph({
          spacing: { before: 20, after: 60, line: 252 },
          children: [
            new TextRun({
              text: "Senior ML/AI Engineer with 7+ years at PayPal designing and deploying production ML and LLM systems for fraud detection, anti-money laundering, and financial risk. Deep expertise in multi-agent LLM orchestration (LangChain, LangGraph), Graph ML (GNN, community detection), and end-to-end model lifecycle management. Proven track record: $100M+ annual fraud loss savings, 85% analyst efficiency gain, 94% SQL generation accuracy. Seeking AI Engineer roles to apply production LLM and ML expertise to next-generation intelligent systems.",
              font: "Calibri",
              size: 18,
            }),
          ],
        }),

        // ── Technical Skills ──
        sectionHeader("TECHNICAL SKILLS"),
        new Paragraph({
          spacing: { before: 20, after: 0, line: 252 },
          children: [
            new TextRun({ text: "LLM & Agents: ", bold: true, font: "Calibri", size: 18 }),
            new TextRun({ text: "LangChain, LangGraph, DSPy, Prompt Engineering, RAG, Graph-RAG, Multi-Agent Systems, Chain-of-Thought", font: "Calibri", size: 18 }),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 0, line: 252 },
          children: [
            new TextRun({ text: "ML & Deep Learning: ", bold: true, font: "Calibri", size: 18 }),
            new TextRun({ text: "PyTorch, DGL, GNN (GAT/GCN), AutoEncoder, LSTM, LightGBM, XGBoost, PU-Learning, Anomaly Detection, SHAP", font: "Calibri", size: 18 }),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 0, line: 252 },
          children: [
            new TextRun({ text: "Data & Infrastructure: ", bold: true, font: "Calibri", size: 18 }),
            new TextRun({ text: "BigQuery, Spark, Neo4j, DGL, Gremlin, Airflow, Docker, GCP, Model Serving", font: "Calibri", size: 18 }),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 60, line: 252 },
          children: [
            new TextRun({ text: "Languages & Tools: ", bold: true, font: "Calibri", size: 18 }),
            new TextRun({ text: "Python, Scala, Java, SQL, Git, Jupyter, PandasAI, Scikit-learn", font: "Calibri", size: 18 }),
          ],
        }),

        // ── Experience ──
        sectionHeader("PROFESSIONAL EXPERIENCE"),
        new Paragraph({
          spacing: { before: 40, after: 0, line: 252 },
          children: [
            new TextRun({ text: "PayPal", bold: true, font: "Calibri", size: 21 }),
            new TextRun({ text: "  \u2014  Senior Machine Learning Engineer", font: "Calibri", size: 20 }),
          ],
        }),
        new Paragraph({
          spacing: { before: 0, after: 20, line: 252 },
          children: [
            new TextRun({ text: "Shanghai, China (Feb 2019 \u2013 Aug 2025)  |  Singapore (Sep 2025 \u2013 Present)", italics: true, font: "Calibri", size: 18, color: "555555" }),
          ],
        }),

        // ── AML Investigation Mate ──
        projectTitle("AML Investigation Mate \u2014 Multi-Agent LLM System", "Jan 2025 \u2013 Present"),
        bullet("Architected a **LangChain deep-agent multi-agent system** (main agent + specialized sub-agents with context isolation) automating end-to-end AML case investigation \u2014 evidence retrieval, regulatory rule reasoning, risk scoring, and SAR report generation \u2014 achieving **80% decision accuracy**, on par with senior investigators and exceeding junior analyst baselines (70\u201375%) on a 100-case benchmark."),
        bullet("Built a **Graph-RAG knowledge layer** (Neo4j) encoding 50+ internal SOPs and external AML regulations into a structured knowledge graph (Condition \u2192 Action \u2192 Legal Document), enabling grounded agent reasoning; reverse-engineered investigator decision logic from historical SAR reports to define sub-agent analytical tools."),
        bullet("Reduced average case review time from **3\u20134 hours to under 30 minutes** (85% reduction) across 15 investigators via **DSPy automatic prompt optimization** and iterative multi-agent evaluation pipelines."),
        bullet("Enabled investigators to **interactively query the agent** for case details, drill into specific evidence, modify analysis parameters, and iteratively refine SAR drafts through a multi-turn conversational interface with full audit trail."),

        // ── Text-to-SQL ──
        projectTitle("Text-to-SQL Investigation Toolkit \u2014 LLM-Powered Query Generation", "Jan 2025 \u2013 Present"),
        bullet("Designed a **multi-step RAG pipeline** (query rephrase \u2192 BGE-M3 embedding retrieval \u2192 Chain-of-Thought SQL generation) converting natural language to BigQuery SQL with **94% correctness** on a 200-query golden dataset, reducing manual SQL coding time by **80%** across 20 investigators."),
        bullet("Built a **stateful multi-agent workflow via LangGraph**, integrating intent routing, error self-healing (3-retry dry-run feedback loop), interactive SQL refinement subgraph, **PandasAI** data visualization, and feedback reuse into a unified human-in-the-loop investigation platform."),

        // ── Graph Fraud Detection ──
        projectTitle("Proactive Fraud Detection \u2014 Graph-Based Account Linking & Clustering", "Jun 2023 \u2013 Dec 2024"),
        bullet("Led end-to-end development of a graph fraud detection pipeline over **100M+ accounts and 300M edges**, evolving through 4 clustering generations (seed-based \u2192 Gremlin multi-hop \u2192 Louvain community detection \u2192 embedding similarity) to surface fraud rings at early account lifecycle \u2014 delivering **$100M+ annual net loss savings**."),
        bullet("Trained **AutoEncoder** via model distillation to extract account embeddings, then applied **edge-aware GAT** (DGL/PyTorch) \u2014 concatenating edge features (linking type, asset riskiness) to attention outputs for full signal preservation \u2014 computing group-level and account-level risk scores; formulated investigator allocation as **Integer Linear Programming (ILP)**."),

        // ── ML Model Development ──
        projectTitle("ML Model Development \u2014 Multi-Domain Risk Modeling", "Feb 2019 \u2013 May 2023"),
        bullet("Owned **end-to-end model lifecycle** across **8+ risk domains** (stolen finance, account takeover, buyer AUP violation, collusion, merchant website compliance, dispute automation), adapting modeling strategies to each domain\u2019s unique constraints \u2014 from tagging design and feature engineering through training, evaluation, and production deployment."),
        bullet("Applied a **wide range of methodologies** tailored to business needs: **PU-Learning** for incomplete-label expansion (95% precision at threshold), **loss-weighted DNN** for high-value fraud prioritization, **LSTM encoder-decoder** for user behavior sequence anomaly detection, **Word2Vec + KMeans** for buyer behavioral clustering, and **LightGBM/XGBoost** as rapid baselines with iterative NN improvements."),
        bullet("Built **merchant website risk scoring** from unstructured HTML and external traffic data; designed **custom feature engineering** pipelines (BIN risk, device fingerprint, IP geo-anomaly, transaction note BERT embeddings); delivered **explainable AI** (SHAP) outputs for stakeholder trust and regulatory compliance across all domains."),

        // ── Leadership ──
        new Paragraph({
          spacing: { before: 50, after: 20, line: 252 },
          children: [
            new TextRun({ text: "Leadership: ", bold: true, font: "Calibri", size: 18 }),
            new TextRun({ text: "Mentored 2 junior data scientists on ML development and fraud domain expertise. Led PayPal\u2019s Recommendation System Study Group (2024), driving cross-team knowledge sharing.", font: "Calibri", size: 18 }),
          ],
        }),

        // ── Education ──
        sectionHeader("EDUCATION"),
        new Paragraph({
          spacing: { before: 40, after: 0, line: 252 },
          children: [
            new TextRun({ text: "The Chinese University of Hong Kong", bold: true, font: "Calibri", size: 19 }),
            new TextRun({ text: "  \u2014  MSc in Business Analytics  |  Sep 2017 \u2013 Nov 2018", font: "Calibri", size: 18 }),
          ],
        }),
        new Paragraph({
          spacing: { before: 20, after: 0, line: 252 },
          children: [
            new TextRun({ text: "College of William & Mary", bold: true, font: "Calibri", size: 19 }),
            new TextRun({ text: "  \u2014  B.S. in Applied Mathematics (CS Minor)  |  Aug 2013 \u2013 May 2017  |  GPA: 3.62/4.0, 6\u00d7 Dean\u2019s List", font: "Calibri", size: 18 }),
          ],
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = "/Users/shenge/gits/interview/resume/ShenGE_resume_enriched.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Enriched resume saved to: " + outPath);
});
