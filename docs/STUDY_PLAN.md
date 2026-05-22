# 4-Week Study Plan — Owning Your Portfolio

> The goal isn't to memorize what's in the repo. The goal is to be able to **whiteboard each project from memory, defend every design decision, and rebuild any component from scratch**. If you can do that, the work is genuinely yours in the only sense that matters to interviewers.

---

## The honest version of "did you build this?"

When an interviewer asks the question you're worried about, the answer that works in 2026 is something like:

> "I built it as a portfolio project, with Claude Code as my pair programmer for scaffolding and boilerplate. The architectural decisions — why LangGraph instead of an LCEL chain, why hybrid retrieval, why full fine-tune on the classifier vs. LoRA on the generator — are mine, and I can defend them. I treated the AI like a senior peer: ask, review, modify, understand. Happy to walk through any specific decision."

This is the answer that gets fresh grads hired right now. The "I built every line myself" answer doesn't survive three follow-up questions and damages trust when it cracks. **Stop trying to hide the tool. Start being the engineer who can defend the work.**

---

## Daily rhythm

Whatever else, every day:
- **~1 hour reading** (paper, blog post, official docs)
- **~1–2 hours coding** (writing, reading repo code, debugging)
- **15 minutes practicing one interview answer out loud**
- **5 minutes journal**: what I learned, what's still fuzzy

Each week has a built-in buffer — if you fall behind, that's normal.

---

## Week 1 — LLM orchestration + RAG (Project 1)

**Goal**: explain Project 1 line-by-line. Understand LangChain 0.3 patterns, LangGraph state machines, hybrid retrieval, evaluation.

### Day 1–2: LangChain 0.3 + LCEL fundamentals
- **Read**: [LangChain Concepts](https://python.langchain.com/docs/concepts/) — the entire concepts section (~2 hours)
- **Watch**: "Build a RAG app in 10 minutes" (any 2024+ video on YouTube — LangChain's official channel is good)
- **Build**: a tiny `prompt | llm | parser` chain in a notebook
- **Drill**: rebuild `01-LLM-RAG/research-assistant/src/llm/prompts.py` from scratch without looking

### Day 3–4: LangGraph state machines
- **Read**: [Why LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) + [LangGraph quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- **Take**: DeepLearning.AI's free short course ["AI Agents in LangGraph"](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) (~1 hour, very practical)
- **Build**: a 3-node graph that does query → retrieve → answer (no reflection yet)
- **Drill**: be able to draw Project 1's graph (rewrite → retrieve → grade → generate → reflect) from memory **on paper** and explain why each node exists

### Day 5: Embeddings + vector stores
- **Read**: [HuggingFace Embeddings course](https://huggingface.co/learn/cookbook/en/index) (find the embeddings cookbook entries)
- **Read**: any "Chroma vs FAISS vs Qdrant" blog post — Pinecone has one
- **Build**: index 5 documents in Chroma, do similarity_search, print the cosine scores
- **Drill**: explain why `sentence-transformers/all-MiniLM-L6-v2` specifically — what's the trade-off if you swap to `bge-large-en-v1.5`?

### Day 6: Hybrid retrieval + reranking
- **Read**: Pinecone's [hybrid search guide](https://docs.pinecone.io/guides/data/understanding-hybrid-search)
- **Read**: Sentence-Transformers docs section on [cross-encoders vs bi-encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- **Build**: extend Day 5's exercise with BM25 + cross-encoder reranking
- **Drill**: explain when BM25 wins (rare keywords), when dense wins (semantic paraphrase), why combining + reranking helps

### Day 7: Project 1 code walkthrough + run it
- Walk through every file in `01-LLM-RAG/research-assistant/src/` — read each one, annotate confusing lines
- Read `docs/ARCHITECTURE.md` and `docs/INTERVIEW_NOTES.md`
- Run the project (locally if possible, Colab if not) — actually answer a real question with it
- Practice the 30-second pitch and the "hardest question" out loud, three times each

---

## Week 2 — Fine-tuning (Project 2)

**Goal**: deeply understand QLoRA. Draw the LoRA math on a whiteboard. Know why `assistant_only_loss=True` matters.

### Day 1: HuggingFace Transformers refresher
- **Read**: [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) — Chapters 1 + 2
- **Drill**: explain what `tokenizer.apply_chat_template()` does and why it's not just string formatting

### Day 2: PyTorch + the Trainer abstraction
- **Read**: HF NLP course Chapter 3 (using `Trainer`)
- **Watch**: any 30-min "PyTorch fundamentals" video if it feels rusty
- **Drill**: explain `bf16` vs `fp16` vs `fp32` and when you'd use each

### Day 3–4: LoRA paper deep dive (the most important reading of the month)
- **Read**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — read twice
- **Read**: [Sebastian Raschka's "Practical Tips for Finetuning LLMs Using LoRA"](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- **Watch**: any "LoRA visualized" video — Yannic Kilcher's explanation is solid
- **Whiteboard practice**: draw `W + (α/r) · BA` and explain why **B is initialized to zero**
- **Build**: implement LoRA on a tiny `nn.Linear` from scratch in pure PyTorch (no `peft` library) — this drill is gold

### Day 5: QLoRA + quantization
- **Read**: [QLoRA paper](https://arxiv.org/abs/2305.14314) — focus on Sections 2 and 3
- **Read**: HuggingFace's [`bitsandbytes` integration docs](https://huggingface.co/docs/transformers/main/en/quantization)
- **Drill**: explain NF4, double-quantization, and what `bnb_4bit_compute_dtype=bfloat16` is doing

### Day 6: TRL + SFTTrainer + chat templates
- **Read**: [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- **Read**: an explanation of SFT vs DPO vs RLHF (TRL's docs have one)
- **Drill**: why does `assistant_only_loss=True` matter for sample efficiency? Why `packing=False`?

### Day 7: Project 2 code walkthrough + run on Colab
- Open Google Colab, request a T4 GPU runtime (free)
- Open `02-NLP-FineTuning/dialogue-summarization/notebooks/01_finetune_samsum_lora.ipynb`
- Run every cell. Watch the loss curve. Save the adapter.
- Push the adapter to your own HF Hub (you'll need an HF write token)
- Read `docs/INTERVIEW_NOTES.md` and answer every question out loud

---

## Week 3 — Classification + Calibration (Project 4) + Dashboards (Project 3)

**Goal**: understand modern encoder fine-tuning, why calibration matters, how to ship a multi-page Streamlit app cleanly.

### Day 1: ModernBERT
- **Read**: [Answer.AI ModernBERT release blog](https://huggingface.co/blog/modernbert) — the whole thing
- **Drill**: explain RoPE vs learned positional embeddings, GeGLU vs GELU, alternating local/global attention

### Day 2: Encoder fine-tuning patterns
- **Read**: HF NLP Course Chapter 3 with classification in mind
- **Drill**: when `AutoModelForSequenceClassification` vs `AutoModelForCausalLM`?
- **Drill**: macro F1 vs micro F1 vs accuracy — when does each matter? With 77 imbalanced classes, why does accuracy lie?

### Day 3–4: Calibration deep dive
- **Read**: [Guo et al. 2017 — On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — at least the intro and the temperature-scaling section
- **Read**: a blog post explaining ECE intuitively (search "expected calibration error explained")
- **Build**: implement temperature scaling from scratch in NumPy. Close `src/calibration.py` first and write it from memory. Then diff against mine.
- **Drill**: what does T > 1 mean physically? What does T = 1 mean? Could T < 1 ever happen?

### Day 5: Streamlit + Plotly
- **Read**: [Streamlit Get Started](https://docs.streamlit.io/get-started) — the entire tutorial
- **Read**: the [Caching section](https://docs.streamlit.io/develop/concepts/architecture/caching) (`st.cache_data` vs `st.cache_resource`)
- **Read**: [Plotly Express quickstart](https://plotly.com/python/plotly-express/)
- **Build**: a 2-page Streamlit app on any small dataset you choose

### Day 6: Project 3 + 4 code walkthroughs
- Walk through `04-Deep-Learning/text-classification-modernbert/`, especially `src/calibration.py` and `src/eval.py`
- Walk through `03-EDA-Dashboards/la-crime-analysis/`, especially the multi-page structure (`pages/` convention) and `src/data.py`
- Read both projects' `docs/INTERVIEW_NOTES.md`

### Day 7: Train + deploy Project 4
- On Colab, run the Project 4 fine-tune (faster than Project 2 — about 15 min)
- Push the model to your HF Hub
- Deploy the Gradio demo to HF Spaces
- Verify the demo actually works end-to-end

---

## Week 4 — Polish + cross-cutting + interview drill

### Day 1: Web stack (for the portfolio site)
- **Read**: MDN on [IntersectionObserver](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API), [Canvas basics](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Basic_animations), [CSS custom properties](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)
- Walk through `portfolio/scripts/neural-bg.js` and `portfolio/scripts/main.js` line-by-line
- **Drill**: explain `requestAnimationFrame` vs `setInterval` — why is RAF better?

### Day 2: Cross-project synthesis
- Read `docs/INTERVIEW_GUIDE.md` cover to cover
- Read `docs/MODERNIZATION_LOG.md` cover to cover
- Make a single-page handwritten cheat-sheet of the cross-project answers ("compare your two fine-tunes", "RAG vs FT", "evaluation philosophy", "how would you scale this")

### Day 3: Mock interview drill
- Record yourself on your phone answering 10 questions from `docs/INTERVIEW_GUIDE.md`
- Play them back. Be honest about where you stumbled.
- Re-record the weak ones.

### Day 4–5: Rebuild a component from scratch (the ownership test)
Pick **one** and do it without looking at the existing code:
- `src/graph/workflow.py` from Project 1 (LangGraph state machine)
- `src/calibration.py` from Project 4 (temperature scaling + ECE)
- `pages/2_Time_Patterns.py` from Project 3 (Streamlit page)

If you can produce a working version using only the official docs as reference, you own the work. If you can't, that's signal about what to study harder.

### Day 6: Polish the surface
- Update LinkedIn with the portfolio link + the 4 projects under "Projects"
- Verify all HF Spaces are deployed and working
- Check the portfolio mobile view on your phone (Lighthouse score it if you want)
- Re-check your CV — does it tell the same story as the portfolio?

### Day 7: Rest
No new material the day before interviews. Sleep matters more than cramming.

---

## Concrete resources

### Papers (priority order — the ones you MUST be able to discuss)
1. **LoRA** — Hu et al. 2021 ([arxiv 2106.09685](https://arxiv.org/abs/2106.09685))
2. **QLoRA** — Dettmers et al. 2023 ([arxiv 2305.14314](https://arxiv.org/abs/2305.14314))
3. **On Calibration of Modern NNs** — Guo et al. 2017 ([arxiv 1706.04599](https://arxiv.org/abs/1706.04599))
4. **BERT** — Devlin et al. 2018 ([arxiv 1810.04805](https://arxiv.org/abs/1810.04805)) — foundation
5. **RAG (original)** — Lewis et al. 2020 ([arxiv 2005.11401](https://arxiv.org/abs/2005.11401))

You don't need to read every paper end-to-end. Read the abstract, intro, and the section relevant to what's in your project.

### Courses (all free)
- [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) — the foundation
- [LangChain Academy](https://academy.langchain.com)
- [DeepLearning.AI Short Course: AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
- [DeepLearning.AI Short Course: Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)
- [DeepLearning.AI Short Course: Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)

### Blogs to bookmark
- [Sebastian Raschka's substack](https://magazine.sebastianraschka.com/) — best practical fine-tuning content on the internet
- [HuggingFace blog](https://huggingface.co/blog)
- [Answer.AI blog](https://www.answer.ai/) (the ModernBERT team)
- [Eugene Yan's blog](https://eugeneyan.com/writing/) — practical applied ML

### Tools to be fluent in
- **Hugging Face Hub** — pushing models, datasets, and deploying Spaces
- **Google Colab** — free T4 GPU, primary training environment
- **Weights & Biases (W&B)** — experiment tracking, used in many real fine-tunes
- **VS Code + Jupyter extension** — your IDE

---

## Technologies the portfolio uses that you may not have studied yet

A specific list, mapped to the project where each one appears:

| Technology | Used in | Status before this plan | After Week N you should be able to defend it |
|---|---|---|---|
| LangChain 0.3 / LCEL | Project 1 | Earlier versions familiar | Week 1 |
| LangGraph (state machines) | Project 1 | Likely new | Week 1 |
| Hybrid retrieval (dense + BM25) | Project 1 | Concept new | Week 1 |
| Cross-encoder reranking | Project 1 | Concept new | Week 1 |
| RAGAS evaluation | Project 1 | New | Week 1 |
| Llama 3.2 / instruction-tuned LLMs | Project 2 | Familiar | Week 2 |
| LoRA (PEFT) | Project 2 | New or shallow | Week 2 |
| QLoRA / `bitsandbytes` | Project 2 | New | Week 2 |
| `trl.SFTTrainer` + `assistant_only_loss` | Project 2 | New | Week 2 |
| Chat templates | Project 2 | Likely shallow | Week 2 |
| ModernBERT | Project 4 | New (Dec 2024 release) | Week 3 |
| Label smoothing | Project 4 | Possibly new | Week 3 |
| Temperature scaling / ECE | Project 4 | Likely new | Week 3 |
| Macro F1, top-K accuracy | Project 4 | Familiar | Week 3 |
| Streamlit multi-page | Project 3 | Likely new convention | Week 3 |
| `st.cache_data` | Project 3 | Likely new | Week 3 |
| Plotly Express + Graph Objects | Project 3 | Possibly familiar | Week 3 |
| Pydantic v2 + `pydantic-settings` | All projects | Possibly new | As-you-go |
| `pyproject.toml` / PEP 621 | All projects | Possibly new | As-you-go |
| Docker multi-stage thinking | All projects | Possibly new | As-you-go |
| Vanilla Canvas API | Portfolio site | Possibly new | Week 4 |
| `IntersectionObserver` | Portfolio site | Possibly new | Week 4 |
| Modern CSS (custom props, backdrop-filter) | Portfolio site | Possibly new | Week 4 |

If a row says "possibly new" and you already know it cold, just skip that day's reading.

---

## How to actually retain this (the boring but true advice)

- **Writing > watching.** After every reading, write a 5-line summary in your own words. The Feynman technique works.
- **Code > read.** After every concept, type out a working example yourself. Mechanical typing puts it in your fingers.
- **Out loud > silent.** After every project, explain it out loud to an imaginary interviewer. Your spoken version reveals exactly what's fuzzy.
- **Sleep > cram.** 7–8 hours per night. Memory consolidation is real and an extra hour of sleep beats an extra hour of cramming.

---

## What "ready" looks like at the end of 4 weeks

You should be able to do all of the following without notes:
- [ ] Draw Project 1's LangGraph state machine on paper and explain each node
- [ ] Whiteboard the LoRA equation `W + (α/r) · BA` and explain why B is initialized to zero
- [ ] Explain what temperature scaling does, why ECE matters, why T > 1 for overconfident models
- [ ] Explain hybrid retrieval and the role of cross-encoder reranking
- [ ] Explain why you used **LoRA** on Project 2 but **full fine-tune** on Project 4
- [ ] Run any of the four projects locally without referring to the README
- [ ] Spend 5 minutes pitching the portfolio without notes

When you can do all of those, the work is yours.
