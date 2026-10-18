# AI Resume Scoring Module (Soft-Skill Evaluation Engine)

> 🔗 **Colab Notebook**: [View the full training, evaluation, and serving pipeline here](https://colab.research.google.com/drive/1XzojQIDxte1NfwP_RuP1lqBH_x1XQV3K?authuser=4#scrollTo=WktVRhxHpNOr)

## Overview
This repository contains the AI evaluation engine for our group's **Applicant Tracking System (ATS)** project, developed in collaboration with **Djezzy**. While the broader system uses a hybrid approach (an ESCO-based taxonomy for technical/hard skills), **my specific responsibility** was designing, testing, and fine-tuning the Small Language Model (SLM).

Instead of relying on generic, unpredictable LLM prompts , I fine-tuned a  model (`Phi-3-mini`) to consistently score candidates against specific soft-skill criteria on a 1–10 scale, outputting strict, parseable JSON for the main Django backend.

---

## The Iterative Journey: Architecture & Constraints

Building a reliable AI evaluator wasn't a one-step process. We went through several iterations to find the optimal architecture, heavily influenced by our hardware and data constraints. 

**Hardware Constraints:** We were restricted to the free tier of Google Colab, utilizing a single **Tesla T4 GPU** (~15GB VRAM). This ruled out massive LLMs and forced us to adopt highly efficient fine-tuning techniques (QLoRA) on compact models.
**Data Constraints:** We had access to two distinct datasets:
1. **Dataset 1 (Small, Rich):** 827 samples. Each contains a resume, a job description, extracted technical (micro) and soft (macro) skills, per-skill matching scores, and textual justifications. Small in size, but highly informative and heavily annotated.
2. **Dataset 2 (Large, Simple):** 6,377 samples. Each provides a resume, a job description, and a single overall matching score (0–100). Lacked per-skill breakdowns or justifications.

### Phase 1: Global Score Regression
- *Approach:* To test the baseline capabilities of the SLM, we fine-tuned it on the larger Dataset 2 to predict a single overall ATS score (0-100).
- *Result:* The model learned to output numbers, but it acted as a complete "black box." Recruiters couldn't see *why* a candidate received a specific score, making it unacceptable for an explainable ATS. Furthermore, the model struggled to grasp the nuanced scoring process from a single scalar value.

### Phase 2: Justification + Global Score
- *Approach:* We switched to the richer Dataset 1 to train the model to generate Chain-of-Thought justifications alongside a global score.
- *Result:* While the justifications were sometimes plausible, the numerical scores became highly unstable. The model was overloaded, trying to learn complex reasoning and regression simultaneously from a limited dataset.

### Phase 3: Per-Skill Scoring (Micro + Macro)
- *Approach:* To prevent overloading the model with a single global score, we tasked it with scoring both technical (micro) and soft (macro) skills, alongside generating justifications.
- *Result:* The model was still overloaded. It had to extract skills, score them, and write coherent justifications using only 827 examples. More importantly, we realized that micro skills (e.g., "Python", "SQL") are concrete and better matched via a deterministic taxonomy, whereas macro skills (e.g., "leadership") require deep contextual understanding. Using a neural approach for both was inefficient and opaque. **This was a major transition point in our project's architecture.**

### Phase 4: The Final Hybrid Architecture (My Module)
- *Approach:* We split the workload. a taxonomy tree handles technical skills. the fine-tuned module strictly handles **soft skills**. Given a resume, a job description, and a list of required soft skills, the model outputs a JSON dictionary of scores (1-10) for *only* the soft skills based on the person's past experiences and achievements.
- *Result:* This focused task perfectly matches the strengths of a Small Language Model (contextual understanding of narratives) while removing the burden of generating explanations or parsing concrete technical skills. The model becomes a specialist for soft-skill assessment, and the small Dataset 1 becomes sufficient because each sample provides multiple macro-skill labels, effectively multiplying the training signal.

---

## Technical Implementation

### 1. Model & Optimization
- **Base Model**: `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters).
- **Optimization**: Used **Unsloth** with **4-bit QLoRA** (quantization + Low-Rank Adaptation).
- **Loss Masking**: Implemented custom data collators to mask the prompt tokens (`-100`). The model *only* calculates loss and updates weights based on the generated `[MACRO_SCORES]` JSON block (averaging only ~3.4% of the sequence), drastically improving training efficiency and focus.

### 2. Engineering for Reliability (FastAPI + Retry Logic)
LLMs occasionally hallucinate or skip requested JSON fields. To prevent the main Django backend from crashing due to incomplete JSON, I built a robust FastAPI microservice:
1. The API receives the resume, JD, and target criteria.
2. It generates the response and parses the `[MACRO_SCORES]` block.
3. **Validation & Retry**: It checks if *all* requested criteria are present in the output. If any are missing, it automatically retries the generation (up to 3 times).
4. It only returns a `422 Unprocessable Entity` error if it fails to produce a complete response after all attempts.
5

---

## Evaluation Results: Base Model vs. Fine-Tuned Model

The base `Phi-3-mini` model suffered from a systematic bias: it tended to blindly overestimate candidate abilities, awarding high scores (7-9) regardless of the actual evidence in the resume. Fine-tuning successfully eliminated this hallucination bias, forcing the model to strictly evaluate the textual evidence based on the candidate's actual past experiences and achievements.

### Example 1: Business Development Manager
*The candidate had basic sales experience but lacked high-level strategic planning depth.*
| Criterion | Ground Truth | Base Model (Phi-3) | Fine-Tuned Model |
| :--- | :---: | :---: | :---: |
| Leadership | 7.0 | 8.0 | **6.0** |
| Business Development | 6.0 | 9.0 | **7.0** |
| Strategic Planning | 5.0 | 7.0 | **5.0** |
> **Observation:** The base model blindly awards high scores. The fine-tuned model accurately penalizes the lack of strategic planning evidence, matching the ground truth exactly.

### Example 2: Senior Business Development Executive
*The candidate's resume was heavily administrative and lacked analytical or leadership depth.*
| Criterion | Ground Truth | Base Model (Phi-3) | Fine-Tuned Model |
| :--- | :---: | :---: | :---: |
| Leadership | 3.0 | 7.0 | **4.0** |
| Business Development | 2.0 | 8.0 | **3.0** |
| Analytical Skills | 2.0 | 9.0 | **2.0** |
> **Observation:** The base model hallucinated high proficiency (7-9) based purely on the job title. The fine-tuned model correctly identified the weak evidence, scoring much closer to the ground truth.

### Example 3: Senior Project Manager
*The candidate had strong, verifiable managerial and client-facing experience.*
| Criterion | Ground Truth | Base Model (Phi-3) | Fine-Tuned Model |
| :--- | :---: | :---: | :---: |
| Leadership | 7.0 | 8.0 | **7.0** |
| Client Relationship | 6.0 | 9.0 | **6.0** |
> **Observation:** The fine-tuned model achieved exact matches by properly weighing the candidate's actual managerial experience without artificially inflating the score.

### Aggregate Performance Metrics
Evaluated on a held-out test set of 63 samples, fine-tuning made a massive difference in reliability and adherence to the scoring rubric:

| Metric | Base Model | Fine-Tuned | Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | 7.3% | **24.1%** | +16.8% |
| **Within ±1 Point Accuracy** | 30.7% | **55.5%** | +24.8% |
| **Mean Absolute Error (MAE)** | 2.36 | **1.42** | -0.94 |
| **Missing Criteria Rate** | 24.8% | **10.9%** | -13.9% |

*Key Takeaway: The fine-tuned model either beat or tied the base model  of individual criterion evaluations, successfully eliminating the base model's tendency to systematically overestimate scores.*
