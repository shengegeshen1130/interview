#!/usr/bin/env python3
"""Generate one-page PDF resumes (English + Chinese) for AI Engineer positions."""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, black
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DARK_GRAY = HexColor("#333333")
MED_GRAY = HexColor("#555555")
LIGHT_GRAY = HexColor("#999999")
ACCENT = HexColor("#2B547E")

# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------

def register_fonts():
    """Register fonts. Use macOS system fonts for Chinese; Helvetica built-in for English."""
    # Chinese fonts
    try:
        pdfmetrics.registerFont(TTFont("Songti", "/System/Library/Fonts/Supplemental/Songti.ttc", subfontIndex=3))
        pdfmetrics.registerFont(TTFont("STHeiti", "/System/Library/Fonts/STHeiti Light.ttc", subfontIndex=0))
        pdfmetrics.registerFont(TTFont("STHeitiBold", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
    except Exception:
        # Fallback to CID font
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


register_fonts()

# ---------------------------------------------------------------------------
# Style factories
# ---------------------------------------------------------------------------

def en_styles(page_w):
    """Return paragraph styles for the English resume."""
    usable = page_w - 2 * 0.5 * inch
    return {
        "name": ParagraphStyle("Name", fontName="Helvetica-Bold", fontSize=18,
                               leading=22, alignment=TA_CENTER, textColor=DARK_GRAY),
        "contact": ParagraphStyle("Contact", fontName="Helvetica", fontSize=9,
                                  leading=12, alignment=TA_CENTER, textColor=MED_GRAY),
        "section": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=10.5,
                                  leading=14, textColor=ACCENT, spaceAfter=2, spaceBefore=1),
        "company": ParagraphStyle("Company", fontName="Helvetica-Bold", fontSize=10,
                                  leading=13, textColor=DARK_GRAY),
        "project": ParagraphStyle("Project", fontName="Helvetica-BoldOblique", fontSize=9.5,
                                  leading=12, textColor=DARK_GRAY),
        "date": ParagraphStyle("Date", fontName="Helvetica-Oblique", fontSize=9,
                               leading=12, alignment=TA_RIGHT, textColor=MED_GRAY),
        "bullet": ParagraphStyle("Bullet", fontName="Helvetica", fontSize=9,
                                 leading=11.5, leftIndent=14, bulletIndent=4,
                                 textColor=DARK_GRAY),
        "keywords": ParagraphStyle("KW", fontName="Helvetica-Oblique", fontSize=7.5,
                                   leading=10, leftIndent=14, textColor=LIGHT_GRAY),
        "skill_cat": ParagraphStyle("SkCat", fontName="Helvetica-Bold", fontSize=8.5,
                                    leading=11.5, textColor=DARK_GRAY),
        "skill_val": ParagraphStyle("SkVal", fontName="Helvetica", fontSize=8.5,
                                    leading=11.5, textColor=DARK_GRAY),
        "edu": ParagraphStyle("Edu", fontName="Helvetica", fontSize=9,
                              leading=11.5, textColor=DARK_GRAY),
        "usable_width": usable,
    }


def zh_styles(page_w):
    """Return paragraph styles for the Chinese resume."""
    usable = page_w - 2 * 0.5 * inch
    body_font = "Songti"
    head_font = "STHeiti"
    bold_font = "STHeitiBold"
    return {
        "name": ParagraphStyle("Name", fontName=bold_font, fontSize=18,
                               leading=22, alignment=TA_CENTER, textColor=DARK_GRAY),
        "contact": ParagraphStyle("Contact", fontName=body_font, fontSize=9,
                                  leading=12, alignment=TA_CENTER, textColor=MED_GRAY),
        "section": ParagraphStyle("Section", fontName=bold_font, fontSize=10.5,
                                  leading=14, textColor=ACCENT, spaceAfter=2, spaceBefore=1),
        "company": ParagraphStyle("Company", fontName=bold_font, fontSize=10,
                                  leading=13, textColor=DARK_GRAY),
        "project": ParagraphStyle("Project", fontName=bold_font, fontSize=9.5,
                                  leading=12, textColor=DARK_GRAY),
        "date": ParagraphStyle("Date", fontName=body_font, fontSize=9,
                               leading=12, alignment=TA_RIGHT, textColor=MED_GRAY),
        "bullet": ParagraphStyle("Bullet", fontName=body_font, fontSize=9,
                                 leading=12, leftIndent=14, bulletIndent=4,
                                 textColor=DARK_GRAY, wordWrap="CJK"),
        "keywords": ParagraphStyle("KW", fontName=body_font, fontSize=7.5,
                                   leading=10, leftIndent=14, textColor=LIGHT_GRAY),
        "skill_cat": ParagraphStyle("SkCat", fontName=bold_font, fontSize=8.5,
                                    leading=11.5, textColor=DARK_GRAY),
        "skill_val": ParagraphStyle("SkVal", fontName=body_font, fontSize=8.5,
                                    leading=11.5, textColor=DARK_GRAY, wordWrap="CJK"),
        "edu": ParagraphStyle("Edu", fontName=body_font, fontSize=9,
                              leading=11.5, textColor=DARK_GRAY, wordWrap="CJK"),
        "usable_width": usable,
        "_body": body_font,
        "_bold": bold_font,
    }

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

EN_CONTENT = {
    "name": "SHEN GE",
    "contact": "1339105051@qq.com  |  +86-18602054952",
    "skills": [
        ("LLM &amp; Agents:", "LangChain, LangGraph, DSPy, RAG, Graph-RAG, Chain-of-Thought, Multi-Agent Systems, Text-to-SQL"),
        ("ML &amp; Graph:", "PyTorch, DGL, GNN (GAT), AutoEncoder, LightGBM, PU-Learning, Anomaly Detection, SHAP"),
        ("Data &amp; Infra:", "Python, SQL, BigQuery, Spark, Neo4j, Gremlin, Pandas, Docker"),
    ],
    "company": "PayPal",
    "title": "Senior Machine Learning Engineer",
    "tenure": "Feb 2019 \u2013 Present",
    "projects": [
        {
            "name": "AML Investigation Mate",
            "date": "Jan 2025 \u2013 Present",
            "bullets": [
                "Architected a <b>multi-agent LLM system</b> (LangChain Deep Agent) with context-isolated sub-agents for evidence retrieval, rule reasoning, and SAR report generation, automating end-to-end AML case investigation and cutting review time from 3\u20134 hours to <b>under 30 minutes (85% reduction)</b>.",
                "Built a <b>Graph-RAG</b> knowledge layer over Neo4j encoding SOPs, regulatory requirements, and historical case outcomes into a structured knowledge graph (SOP \u2192 Condition \u2192 Action \u2192 Legal Reference) for grounded agent reasoning.",
                "Achieved <b>80% decision accuracy</b> (vs. 70\u201375% for junior investigators) on a 100-case benchmark via <b>DSPy-based automatic prompt optimization</b> and iterative agent evaluation with reverse-engineered investigator decision logic from historical SAR reports.",
            ],
            "keywords": "Multi-Agent | LangChain Deep Agent | Graph-RAG | Neo4j | DSPy | Prompt Optimization",
        },
        {
            "name": "Text-to-SQL for Risk Investigation",
            "date": "Jan 2025 \u2013 Present",
            "bullets": [
                "Designed an LLM-powered NL-to-SQL engine with multi-step <b>RAG + Chain-of-Thought</b> prompting (query rephrase \u2192 BGE-M3 embedding \u2192 few-shot retrieval \u2192 SQL generation), achieving <b>94% SQL correctness</b> (188/200 golden queries) and <b>80% reduction</b> in manual query time.",
                "Built a stateful multi-agent workflow via <b>LangGraph</b> with isolated subgraphs for intent routing, SQL editing, data visualization (PandasAI), and a feedback loop that continuously enriches the vector DB with validated query\u2013SQL pairs.",
                "Implemented <b>BigQuery dry-run validation</b> with an error-feedback retry mechanism (up to 3 rounds), enabling self-correcting SQL generation and supporting 20+ investigators in daily case investigation workflows.",
            ],
            "keywords": "Text-to-SQL | LangGraph | BigQuery | RAG | CoT | BGE-M3 | PandasAI | Human-in-the-Loop",
        },
        {
            "name": "Proactive Trend Detection with Account Linking &amp; Clustering",
            "date": "Jun 2023 \u2013 Dec 2024",
            "bullets": [
                "Led from 0-to-1 a graph-based fraud detection system achieving <b>$100M+ annual net loss saving</b>, constructing account graphs with <b>100M+ nodes and 200\u2013300M edges</b> from credit card, device, IP, and transaction linking signals.",
                "Designed a multi-strategy clustering pipeline (seed-based \u2192 Gremlin multi-hop traversal \u2192 Louvain community detection \u2192 embedding cosine similarity) to maximize recall across diverse fraud patterns, analogous to multi-recall in recommendation systems.",
                "Built an <b>edge-aware GAT</b> that concatenates edge features (link type, asset riskiness) into node representations, combined with <b>AutoEncoder</b>-distilled account embeddings, for joint group- and account-level risk scoring; formulated resource allocation as <b>ILP</b> for cluster deduplication.",
            ],
            "keywords": "GNN | GAT | AutoEncoder | Community Detection | ILP | PyTorch | DGL | Gremlin | Spark | BigQuery",
        },
        {
            "name": "ML Model Development \u2014 Multiple Risk Domains",
            "date": "Feb 2019 \u2013 May 2023",
            "bullets": [
                "Owned <b>end-to-end model lifecycle</b> (problem framing \u2192 tagging analysis \u2192 feature engineering \u2192 training \u2192 deployment) across 6+ risk domains including stolen finance, account takeover, merchant collusion, and dispute automation.",
                "Applied <b>PU-Learning</b> to expand incomplete fraud labels (95% precision at threshold), <b>Word2Vec</b> buyer embeddings to capture behavioral network patterns, and <b>loss-weighted training</b> to prioritize high-value fraud detection.",
                "Built sequence-based <b>anomaly detection</b> models and <b>SHAP</b>-driven explainability dashboards; iteratively improved models via tagging \u2192 baseline \u2192 feature iteration \u2192 tuning cycles using LightGBM and DNN architectures.",
            ],
            "keywords": "LightGBM | DNN | PU-Learning | Word2Vec | Anomaly Detection | SHAP | Feature Engineering | Model Serving",
        },
    ],
    "leadership": "Mentored 2 junior data scientists on ML development and fraud domain knowledge; led PayPal\u2019s cross-team Recommendation System Study Group driving knowledge sharing on state-of-the-art RecSys approaches (2024).",
    "education": [
        "<b>MSc Business Analytics</b>, Chinese University of Hong Kong (2017\u20132018)",
        "<b>BS Applied Mathematics</b> (CS Minor), College of William &amp; Mary (2013\u20132017) \u2014 GPA: 3.62/4.0, 6\u00d7 Dean\u2019s List",
    ],
}

ZH_CONTENT = {
    "name": "SHEN GE",
    "contact": "1339105051@qq.com  |  +86-18602054952",
    "skills": [
        ("LLM &amp; Agents:", "LangChain, LangGraph, DSPy, RAG, Graph-RAG, Chain-of-Thought, Multi-Agent Systems, Text-to-SQL"),
        ("ML &amp; Graph:", "PyTorch, DGL, GNN (GAT), AutoEncoder, LightGBM, PU-Learning, Anomaly Detection, SHAP"),
        ("Data &amp; Infra:", "Python, SQL, BigQuery, Spark, Neo4j, Gremlin, Pandas, Docker"),
    ],
    "section_skills": "\u6280\u672f\u6280\u80fd (Technical Skills)",
    "section_exp": "\u5de5\u4f5c\u7ecf\u5386 (Work Experience)",
    "section_edu": "\u6559\u80b2\u80cc\u666f (Education)",
    "company": "PayPal",
    "title": "\u9ad8\u7ea7\u673a\u5668\u5b66\u4e60\u5de5\u7a0b\u5e08 (Senior ML Engineer)",
    "tenure": "2019\u5e742\u6708 \u2013 \u81f3\u4eca",
    "projects": [
        {
            "name": "AML\u667a\u80fd\u8c03\u67e5\u7cfb\u7edf (AML Investigation Mate)",
            "date": "2025\u5e741\u6708 \u2013 \u81f3\u4eca",
            "bullets": [
                "\u57fa\u4e8e <b>LangChain Deep Agent</b> \u642d\u5efa\u591aAgent\u67b6\u6784\uff0cContext\u9694\u79bb\u7684\u5b50Agent\u5206\u522b\u8d1f\u8d23\u8bc1\u636e\u68c0\u7d22\u3001\u89c4\u5219\u63a8\u7406\u548cSAR\u62a5\u544a\u751f\u6210\uff0c\u5c06\u53cd\u6d17\u94b1\u6848\u4ef6\u8c03\u67e5\u5168\u6d41\u7a0b\u81ea\u52a8\u5316\uff0c\u5355\u6848\u5ba1\u67e5\u65f6\u95f4\u4ece3\u20134\u5c0f\u65f6\u964d\u81f3<b>30\u5206\u949f\u4ee5\u5185\uff08\u63d0\u6548\u7ea685%\uff09</b>\u3002",
                "\u642d\u5efa <b>Graph-RAG</b> \u77e5\u8bc6\u5c42\uff08Neo4j\uff09\uff0c\u5c06\u5185\u90e8SOP\u3001\u76d1\u7ba1\u8981\u6c42\u548c\u5386\u53f2\u6848\u4f8b\u7ed3\u6784\u5316\u4e3a\u77e5\u8bc6\u56fe\u8c31\uff08SOP \u2192 Condition \u2192 Action \u2192 Legal Reference\uff09\uff0c\u4e3aAgent\u63a8\u7406\u63d0\u4f9b\u53ef\u8ffd\u6eaf\u7684\u4f9d\u636e\u3002",
                "\u901a\u8fc7\u53cd\u5411\u5206\u6790\u5386\u53f2SAR\u62a5\u544a\u63a8\u7406\u8c03\u67e5\u5458\u51b3\u7b56\u903b\u8f91\uff0c\u7ed3\u5408<b>DSPy\u81ea\u52a8Prompt\u4f18\u5316</b>\uff0c\u5728100\u4e2a\u6d4b\u8bd5Case\u4e0a\u8fbe\u5230<b>80%\u51b3\u7b56\u51c6\u786e\u7387</b>\uff08\u521d\u7ea7\u8c03\u67e5\u545870\u201375%\uff09\u3002",
            ],
            "keywords": "Multi-Agent | LangChain Deep Agent | Graph-RAG | Neo4j | DSPy | Prompt Optimization",
        },
        {
            "name": "\u81ea\u7136\u8bed\u8a00\u8f6cSQL\u8c03\u67e5\u5de5\u5177 (Text-to-SQL)",
            "date": "2025\u5e741\u6708 \u2013 \u81f3\u4eca",
            "bullets": [
                "\u8bbe\u8ba1LLM\u9a71\u52a8\u7684NL-to-SQL\u5f15\u64ce\uff0c\u91c7\u7528\u591a\u6b65<b>RAG + Chain-of-Thought</b>\uff08Query Rephrase \u2192 BGE-M3 Embedding \u2192 Few-shot\u68c0\u7d22 \u2192 SQL\u751f\u6210\uff09\uff0c\u5728200\u6761Golden Dataset\u4e0a\u8fbe\u5230<b>94% SQL\u6b63\u786e\u7387</b>\uff0c\u4eba\u5de5\u5199SQL\u65f6\u95f4<b>\u51cf\u5c1180%</b>\u3002",
                "\u57fa\u4e8e <b>LangGraph</b> \u642d\u5efa\u6709\u72b6\u6001\u591aAgent\u5de5\u4f5c\u6d41\uff0c\u7528\u72ec\u7acbSubgraph\u5904\u7406\u610f\u56fe\u8def\u7531\u3001SQL\u7f16\u8f91\u3001\u6570\u636e\u53ef\u89c6\u5316\uff08PandasAI\uff09\uff0c\u5e76\u901a\u8fc7Feedback Loop\u6301\u7eed\u5c06\u9a8c\u8bc1\u901a\u8fc7\u7684Query-SQL Pair\u5165\u5e93\u6269\u5145Vector DB\u3002",
                "\u5b9e\u73b0<b>BigQuery Dry Run\u9a8c\u8bc1</b>\u4e0e\u9519\u8bef\u53cd\u9988\u91cd\u8bd5\u673a\u5236\uff08\u6700\u591a3\u8f6e\uff09\uff0c\u652f\u6301\u81ea\u7ea0\u9519SQL\u751f\u6210\uff0c\u670d\u52a120+\u8c03\u67e5\u5458\u7684\u65e5\u5e38Case\u8c03\u67e5\u5de5\u4f5c\u6d41\u3002",
            ],
            "keywords": "Text-to-SQL | LangGraph | BigQuery | RAG | CoT | BGE-M3 | PandasAI | Human-in-the-Loop",
        },
        {
            "name": "\u4e3b\u52a8\u6b3a\u8bc8\u68c0\u6d4b\u4e0e\u8d26\u6237\u805a\u7c7b (Proactive Trend Detection)",
            "date": "2023\u5e746\u6708 \u2013 2024\u5e7412\u6708",
            "bullets": [
                "\u4ece0\u52301\u4e3b\u5bfc\u57fa\u4e8e\u56fe\u7684\u6b3a\u8bc8\u68c0\u6d4b\u7cfb\u7edf\uff0c\u5e74\u5316<b>\u51c0\u635f\u5931\u6321\u83b7\u8d85$100M</b>\uff1b\u6784\u5efa<b>1\u4ebf+\u8282\u70b9\u30012\u20133\u4ebf\u6761\u8fb9</b>\u7684\u8d26\u6237\u5173\u8054\u56fe\uff08\u4fe1\u7528\u5361\u3001\u8bbe\u5907\u3001IP\u3001\u4ea4\u6613\u7b49Linking Signal\uff09\u3002",
                "\u8bbe\u8ba1\u591a\u7b56\u7565\u805a\u7c7bPipeline\uff08Seed-based \u2192 Gremlin\u591a\u8df3\u904d\u5386 \u2192 Louvain\u793e\u533a\u53d1\u73b0 \u2192 Embedding\u4f59\u5f26\u76f8\u4f3c\u5ea6\uff09\uff0c\u7c7b\u4f3c\u63a8\u8350\u7cfb\u7edf\u591a\u8def\u53ec\u56de\uff0c\u6700\u5927\u5316\u4e0d\u540c\u6b3a\u8bc8\u6a21\u5f0f\u7684Recall\u3002",
                "\u642d\u5efa<b>Edge-aware GAT</b>\uff0c\u5c06\u8fb9\u7279\u5f81\uff08Linking\u7c7b\u578b\u3001Asset\u98ce\u9669\u5ea6\uff09Concat\u5230\u8282\u70b9\u8868\u793a\u4e2d\uff0c\u7ed3\u5408<b>AutoEncoder</b>\u84b8\u998f\u7684\u8d26\u6237Embedding\uff0c\u505a\u7ec4\u7ea7+\u8d26\u6237\u7ea7\u53cc\u5c42\u98ce\u9669\u6253\u5206\uff1b\u7528<b>ILP</b>\u5bf9\u6b3a\u8bc8\u7c07\u53bb\u91cd\u5e76\u4f18\u5316\u8c03\u67e5\u5458\u8d44\u6e90\u5206\u914d\u3002",
            ],
            "keywords": "GNN | GAT | AutoEncoder | Community Detection | ILP | PyTorch | DGL | Gremlin | Spark | BigQuery",
        },
        {
            "name": "\u591a\u98ce\u63a7\u9886\u57dfML\u6a21\u578b\u5f00\u53d1 (ML Model Development)",
            "date": "2019\u5e742\u6708 \u2013 2023\u5e745\u6708",
            "bullets": [
                "\u8d1f\u8d23<b>\u6a21\u578b\u5168\u751f\u547d\u5468\u671f</b>\uff08\u95ee\u9898\u5b9a\u4e49 \u2192 Tagging\u5206\u6790 \u2192 \u7279\u5f81\u5de5\u7a0b \u2192 \u8bad\u7ec3 \u2192 \u90e8\u7f72\uff09\uff0c\u8986\u76d66+\u4e2a\u98ce\u63a7\u65b9\u5411\uff1a\u76d7\u5237\u3001\u8d26\u6237\u76d7\u7528\u3001\u5546\u6237\u4e32\u8c0b\u3001\u4e89\u8bae\u81ea\u52a8\u5316\u7b49\u3002",
                "\u7528<b>PU-Learning</b>\u6269\u5145\u4e0d\u5b8c\u6574\u6b3a\u8bc8\u6807\u7b7e\uff08\u9608\u503c\u4e0a95%\u7cbe\u786e\u7387\uff09\uff0c<b>Word2Vec</b> Buyer Embedding\u6355\u6349\u884c\u4e3a\u7f51\u7edc\u6a21\u5f0f\uff0c<b>Loss-weighted Training</b>\u8ba9\u6a21\u578b\u4f18\u5148\u5173\u6ce8\u9ad8\u635f\u5931\u4ea4\u6613\u3002",
                "\u642d\u5efa\u5e8f\u5217<b>\u5f02\u5e38\u68c0\u6d4b</b>\u6a21\u578b\u548c<b>SHAP</b>\u53ef\u89e3\u91ca\u6027\u770b\u677f\uff1b\u901a\u8fc7Tagging \u2192 Baseline \u2192 Feature\u8fed\u4ee3 \u2192 \u6a21\u578b\u8c03\u4f18\u7684\u5faa\u73af\u6301\u7eed\u6539\u8fdb\uff0c\u4f7f\u7528LightGBM\u548cDNN\u67b6\u6784\u3002",
            ],
            "keywords": "LightGBM | DNN | PU-Learning | Word2Vec | Anomaly Detection | SHAP | Feature Engineering | Model Serving",
        },
    ],
    "leadership": "\u5e26\u65592\u540d\u521d\u7ea7\u6570\u636e\u79d1\u5b66\u5bb6\uff0c\u52a0\u901f\u5176ML\u5f00\u53d1\u548c\u98ce\u63a7\u4e1a\u52a1\u4e0a\u624b\uff1b\u7ec4\u7ec7PayPal\u8de8\u56e2\u961f\u63a8\u8350\u7cfb\u7edf\u5b66\u4e60\u5c0f\u7ec4\uff0c\u63a8\u52a8RecSys\u524d\u6cbf\u6280\u672f\u5206\u4eab\uff082024\uff09\u3002",
    "education": [
        "<b>\u5546\u4e1a\u5206\u6790\u7855\u58eb (MSc Business Analytics)</b>\uff0c\u9999\u6e2f\u4e2d\u6587\u5927\u5b66 (2017\u20132018)",
        "<b>\u5e94\u7528\u6570\u5b66\u5b66\u58eb (BS Applied Mathematics)</b>\uff0c\u8f85\u4fee CS\uff0cCollege of William &amp; Mary (2013\u20132017) \u2014 GPA: 3.62/4.0\uff0c6\u00d7 Dean\u2019s List",
    ],
}

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC"),
                      spaceBefore=4, spaceAfter=4)


def _skill_row(cat, val, styles):
    """Return a Table row for a skill category."""
    cat_p = Paragraph(cat, styles["skill_cat"])
    val_p = Paragraph(val, styles["skill_val"])
    t = Table([[cat_p, val_p]], colWidths=[95, styles["usable_width"] - 95])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _project_header(name, date, styles):
    """Project name left, date right on one line."""
    left = Paragraph(name, styles["project"])
    right = Paragraph(date, styles["date"])
    w = styles["usable_width"]
    t = Table([[left, right]], colWidths=[w * 0.72, w * 0.28])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def build_resume(content, styles, page_size, output_path, lang="en"):
    """Build a one-page PDF resume."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=page_size,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.4 * inch, bottomMargin=0.4 * inch,
    )

    story = []

    # --- Name ---
    story.append(Paragraph(content["name"], styles["name"]))
    story.append(Spacer(1, 2))

    # --- Contact ---
    story.append(Paragraph(content["contact"], styles["contact"]))
    story.append(_hr())

    # --- Technical Skills ---
    section_label = content.get("section_skills", "TECHNICAL SKILLS")
    story.append(Paragraph(section_label, styles["section"]))
    for cat, val in content["skills"]:
        story.append(_skill_row(cat, val, styles))
    story.append(Spacer(1, 1))
    story.append(_hr())

    # --- Work Experience ---
    section_label = content.get("section_exp", "WORK EXPERIENCE")
    story.append(Paragraph(section_label, styles["section"]))

    # Company + title line
    company_line = f'{content["company"]} \u2014 {content["title"]}'
    tenure = content["tenure"]
    left = Paragraph(company_line, styles["company"])
    right = Paragraph(tenure, styles["date"])
    w = styles["usable_width"]
    t = Table([[left, right]], colWidths=[w * 0.72, w * 0.28])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(t)
    story.append(Spacer(1, 3))

    # Projects
    for proj in content["projects"]:
        story.append(_project_header(proj["name"], proj["date"], styles))
        for bullet_text in proj["bullets"]:
            story.append(Paragraph(bullet_text, styles["bullet"],
                                   bulletText="\u2022"))
        story.append(Paragraph(proj["keywords"], styles["keywords"]))
        story.append(Spacer(1, 3))

    # Leadership
    leader_label = "Leadership: " if lang == "en" else "\u56e2\u961f\u7ba1\u7406: "
    if lang == "zh":
        leader_text = f'<font name="{styles["_bold"]}">{leader_label}</font>{content["leadership"]}'
    else:
        leader_text = f"<b>{leader_label}</b>{content['leadership']}"
    story.append(Paragraph(leader_text, styles["bullet"]))
    story.append(_hr())

    # --- Education ---
    section_label = content.get("section_edu", "EDUCATION")
    story.append(Paragraph(section_label, styles["section"]))
    for edu_line in content["education"]:
        story.append(Paragraph(edu_line, styles["edu"]))

    # Build
    doc.build(story)
    print(f"Generated: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    en_path = os.path.join(OUTPUT_DIR, "ShenGE_resume_EN.pdf")
    zh_path = os.path.join(OUTPUT_DIR, "ShenGE_resume_ZH.pdf")

    build_resume(EN_CONTENT, en_styles(letter[0]), letter, en_path, lang="en")
    build_resume(ZH_CONTENT, zh_styles(A4[0]), A4, zh_path, lang="zh")
