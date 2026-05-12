# Design: OKX Anti-Fraud Interview Prep — Transformer-Based Fraud Detection

**Date:** 2026-05-12  
**Target role:** Senior Staff AI Engineer, Anti-Fraud — OKX  
**Output location:** `interview_question/okx_anti_fraud/`

---

## Context

The user is preparing for an OKX interview for a role focused on transformer-based ML models for blockchain fraud detection. Key JD requirements: transformer architectures (BERT, GPT), graph databases (Neo4j), LangGraph agents, PyTorch/Hugging Face, on-chain data.

**User profile:** Working knowledge of transformers (has used BERT/GPT in projects, needs deeper treatment of attention math and training tricks). New to blockchain transaction data structures.

---

## Output: 4 Files in `interview_question/okx_anti_fraud/`

### File 1: `01_blockchain_data_primer.md`

**Purpose:** Build the blockchain data foundation before any ML content.

**Sections:**
1. What is a blockchain — ledger, blocks, transactions, consensus basics (ML context only)
2. Transaction data anatomy — inputs/outputs, addresses, amounts, timestamps, gas fees; UTXO model (Bitcoin) vs. account model (Ethereum)
3. On-chain entities — wallets, smart contracts, token transfers (ERC-20/ERC-721), DEX swaps
4. Blockchain data as a dataset — tabular features, transaction graphs (nodes = addresses, edges = transfers), time-series view
5. Fraud patterns in blockchain — money laundering (mixing, layering), Ponzi schemes, phishing wallets, flash loan attacks, wash trading
6. Interview Q&A — ~5 questions

---

### File 2: `02_transformer_architecture.md`

**Purpose:** Solidify transformer math and design choices from working knowledge to interview-depth.

**Sections:**
1. Why transformers — RNN/LSTM limitations, what attention solves
2. Scaled dot-product attention — full math, intuition behind Q/K/V, scaling factor
3. Multi-head attention — multiple heads, concatenation, what different heads learn
4. Positional encoding — sinusoidal, learned, RoPE, ALiBi; permutation-invariance problem
5. Full transformer block — Layer Norm (pre vs. post), FFN (4× expansion), residual connections, dropout
6. Encoder vs. decoder vs. encoder-decoder — masked self-attention, when to use each
7. Training details — MLM, CLM, span corruption; fine-tuning; O(n²) complexity implications
8. Interview Q&A — ~8 questions

---

### File 3: `03_transformer_variants.md`

**Purpose:** Cover the landscape of transformer variants with fraud-relevance highlighted for each.

**Sections:**
1. BERT family — BERT, RoBERTa, DeBERTa, ALBERT; use case differences
2. GPT family — GPT-1/2/3, causal LM, in-context learning; GPT vs. BERT use cases
3. Efficient transformers — Longformer, BigBird, Performer; solving O(n²) for long sequences
4. Tabular transformers — TabTransformer, FTTransformer; modeling fraud feature tables
5. Time-series transformers — TFT, PatchTST, iTransformer; modeling transaction sequences
6. Anomaly detection variants — Anomaly Transformer, TranAD; directly applicable to fraud
7. Graph + Transformer hybrids — Graphormer, Graph Transformer; blockchain graph signals
8. Comparison table — architecture × use case × fraud relevance
9. Interview Q&A — ~8 questions

---

### File 4: `04_fraud_detection_with_transformers.md`

**Purpose:** End-to-end system design: how to actually build a transformer-based fraud detection system on blockchain data.

**Sections:**
1. Problem framing — anomaly detection vs. classification; label scarcity, class imbalance, adversarial drift
2. Feature engineering — address-level, transaction sequence, graph-derived (degree, PageRank), temporal features
3. Approach 1: Sequence modeling — BERT-style pre-training on tx sequences, fine-tuning on fraud labels
4. Approach 2: Tabular transformer — FTTransformer/TabTransformer for supervised classification
5. Approach 3: Graph + Transformer — GNN embeddings fed into transformer; multi-hop laundering
6. Approach 4: Anomaly detection — unsupervised/semi-supervised with Anomaly Transformer or TranAD
7. Production concerns — concept drift, online learning, explainability (attention weights), latency
8. OKX-specific angle — DeFi fraud (flash loans, rug pulls, wash trading), blockchain graph richness vs. traditional finance
9. Interview Q&A — ~10 questions

---

## Format Conventions

- Follow existing repo Q&A format: `### Q{N}: {question}` → `**回答：**` → numbered points → `> **Follow-up 提示：**`
- Section titles bilingual: Chinese + English in parentheses
- Technical terms in inline code, math in LaTeX with `_{...}` subscript notation
- Markdown tables for comparisons

---

## Success Criteria

- User can answer any transformer architecture question at interview depth
- User can explain 3+ approaches to applying transformers to blockchain fraud
- User understands blockchain data well enough to discuss feature engineering confidently
- Materials are self-contained — no prior blockchain knowledge required to read File 1 → File 4 in order
