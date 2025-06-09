
---

# Multimodal RAG System Evaluation

## Overview
This repository contains the evaluation results of a Multimodal Retrieval-Augmented Generation (RAG) System, designed to process queries and retrieve relevant articles and images using advanced language models (e.g., Gemini and Groq agents). The system integrates query preprocessing, text and image retrieval engines, and response generation, evaluated based on binary correctness and detailed metric scores.

## Evaluation Metrics

### Binary Accuracy Metrics
The system was assessed for the correctness of its responses, where `1` indicates a correct answer and `0` indicates an incorrect answer. The results are as follows:

- **Total Responses**: 102
- **Correct Count**: 69
- **Incorrect Count**: 31
- **Accuracy**: 0.69 (69%)
- **Correct Percentage**: 69.0%
- **Incorrect Percentage**: 31.0%
- **Imbalance Ratio (Correct/Incorrect)**: 2.22580451619303

**Analysis**: The system achieves a 69% accuracy rate, indicating that nearly 7 out of 10 responses are correct. The imbalance ratio of 2.23 suggests a moderate skew toward correct answers, with about twice as many correct responses as incorrect ones.

### Detailed Metric Scores
Each response was evaluated across four metrics: `faithfulness`, `relevance`, `groundedness`, and `helpfulness`, scored on a scale of 1 to 5. The aggregate statistics are:

- **Overall Average Score**: 4.30
  - Reflects the mean score across all metrics and responses, indicating a strong overall performance.

#### Faithfulness
- **Mean**: 4.73
- **Median**: 5.00
- **Standard Deviation**: 0.74
- **Minimum**: 1
- **Maximum**: 5
- **Distribution**: {4: 12, 5: 85, 1: 1, 2: 4}

**Analysis**: High faithfulness (mean 4.73) suggests responses are largely consistent with the source material, with 85 responses scoring the maximum of 5.

#### Relevance
- **Mean**: 4.36
- **Median**: 5.00
- **Standard Deviation**: 1.33
- **Minimum**: 1
- **Maximum**: 5
- **Distribution**: {2: 2, 5: 80, 3: 6, 1: 11, 4: 3}

**Analysis**: A mean of 4.36 indicates good relevance, though the higher standard deviation (1.33) and presence of low scores (e.g., 11 scores of 1) suggest inconsistency in addressing query intent.

#### Groundedness
- **Mean**: 3.77
- **Median**: 4.00
- **Standard Deviation**: 1.24
- **Minimum**: 1
- **Maximum**: 5
- **Distribution**: {4: 40, 2: 9, 3: 10, 5: 34, 1: 9}

**Analysis**: A mean of 3.77 with a balanced distribution (e.g., 34 scores of 5, 40 of 4) shows solid grounding in facts, though some responses score as low as 1.

#### Helpfulness
- **Mean**: 4.53
- **Median**: 5.00
- **Standard Deviation**: 1.15
- **Minimum**: 1
- **Maximum**: 5
- **Distribution**: {3: 8, 5: 70, 4: 12, 1: 4, 2: 8}

**Analysis**: A high mean of 4.53 and 70 scores of 5 indicate strong helpfulness, with minimal low scores (4 scores of 1).

## Analysis
- **Strengths**:
  - The system excels in `faithfulness` (4.73) and `helpfulness` (4.53), suggesting responses are accurate and useful to users.
  - An overall average score of 4.30 across all metrics reflects robust performance.
  - Binary accuracy (69%) indicates a reliable baseline for correct answers.

- **Weaknesses**:
  - `Relevance` shows the highest variability (std = 1.33) and includes 11 scores of 1, indicating some responses fail to address the query intent.
  - `Groundedness` (mean 3.77) is the lowest metric, with a spread of scores (1 to 5), suggesting occasional lack of factual grounding.

- **Trends**:
  - The distribution of scores (e.g., 80–85% of `faithfulness` and `helpfulness` at 5) shows a strong positive skew, while `relevance` and `groundedness` have more spread, indicating areas for improvement.


## Setup and Usage
1. **Dependencies**: Ensure the following are installed:
   - `python>=3.9`
   - `gradio`, `python-dotenv`, `sentence-transformers`, `faiss-cpu`, `langchain-groq`, `langchain-google-genai`
   - Install via: `pip install -r requirements.txt` (create a `requirements.txt` with listed packages).
2. **Environment**: Set `GOOGLE_API_KEY` and `GROQ_API_KEY` in a `.env` file.
3. **Data**: Place FAISS indices and metadata in `data/faiss/` and `data/metadata/` directories.
4. **Run**: Execute `python main.py` to launch the system and evaluate new responses.

## Contributing
Contributions are welcome! Please open issues for bugs or suggestions and submit pull requests with improvements. Focus areas include enhancing relevance and groundedness based on the evaluation results.

---

