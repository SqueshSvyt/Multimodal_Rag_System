

questions = [
    "What is artificial intelligence?",
    "How does machine learning differ from traditional programming?",
    "What are the main types of machine learning?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is reinforcement learning?",
    "Who coined the term artificial intelligence?",
    "What was the Dartmouth Conference?",
    "What is deep learning?",
    "How do neural networks work?",
    "What is a convolutional neural network (CNN)?",
    "What are recurrent neural networks (RNNs) used for?",
    "What is transfer learning in AI?",
    "How does GPT work?",
    "What is a transformer architecture in AI?",
    "What are attention mechanisms in neural networks?",
    "What is the role of data in machine learning?",
    "Why is data preprocessing important in AI?",
    "What is overfitting in machine learning?",
    "How can overfitting be prevented?",
    "What is underfitting in machine learning?",
    "What is the bias-variance tradeoff?",
    "What are common evaluation metrics in classification problems?",
    "What is precision and recall in AI?",
    "What is the F1 score?",
    "What is cross-validation?",
    "What is hyperparameter tuning?",
    "What are decision trees?",
    "What is the difference between bagging and boosting?",
    "What is random forest?",
    "What is gradient boosting?",
    "How does XGBoost work?",
    "What is support vector machine (SVM)?",
    "What is k-means clustering?",
    "What is dimensionality reduction?",
    "What is principal component analysis (PCA)?",
    "What is t-SNE used for?",
    "How does natural language processing (NLP) work?",
    "What is named entity recognition (NER)?",
    "What is sentiment analysis in AI?",
    "What are embeddings in NLP?",
    "What is Word2Vec?",
    "How does BERT model work?",
    "What is the difference between GPT and BERT?",
    "What is zero-shot learning?",
    "What is few-shot learning?",
    "What is the Turing Test?",
    "Who was Alan Turing?",
    "What are AI hallucinations?",
    "How can AI be used in healthcare?",
    "What are the risks of using AI in critical systems?",
    "What is explainable AI (XAI)?",
    "Why is AI ethics important?",
    "What are the concerns around bias in AI models?",
    "How can AI models be audited?",
    "What is model interpretability?",
    "What is fairness in machine learning?",
    "What are the societal implications of AI?",
    "What are deepfakes?",
    "What are generative adversarial networks (GANs)?",
    "How do GANs work?",
    "What are autonomous vehicles?",
    "How does AI power recommendation systems?",
    "What are AI agents?",
    "What is multi-agent reinforcement learning?",
    "What is federated learning?",
    "How does AI impact privacy?",
    "What is differential privacy in AI?",
    "What is continual learning?",
    "What is meta-learning?",
    "What are AI benchmarks?",
    "What is the difference between narrow AI and general AI?",
    "What is artificial general intelligence (AGI)?",
    "What is the singularity in AI?",
    "What is the role of AI in robotics?",
    "What is computer vision?",
    "What is object detection?",
    "What is semantic segmentation in AI?",
    "What is pose estimation?",
    "How does facial recognition work?",
    "What are ethical issues in facial recognition?",
    "What are AI-powered chatbots?",
    "What is prompt engineering?",
    "What are temperature and top-p in language models?",
    "How can AI help fight climate change?",
    "What are some limitations of current AI systems?",
    "What is multimodal AI?",
    "What is Retrieval-Augmented Generation (RAG)?",
    "How does a RAG system work?",
    "What are the components of a RAG pipeline?",
    "How to evaluate a RAG system?",
    "What are embeddings used for in RAG?",
    "How does document retrieval work in RAG?",
    "What models are commonly used in RAG?",
    "What are hallucinations in RAG models?",
    "What are some applications of RAG systems?",
    "What is the future of RAG in AI?",
    "What is the difference between extractive and generative QA?",
    "What datasets are used to train QA systems?",
    "What is the role of vector databases in RAG?",
    "What is cosine similarity in document retrieval?",
    "What are knowledge-grounded responses in AI?"
]

# Step 1: Create 100 AI-related questions
questions_type2 = [
    f"What is artificial intelligence and how does it work?" if i == 0 else
    f"Explain the concept of AI in the context of {topic}."
    for i, topic in enumerate([
        "machine learning", "deep learning", "natural language processing", "computer vision", "autonomous vehicles",
        "robotics", "healthcare", "education", "finance", "cybersecurity", "ethics", "explainability", "AI alignment",
        "AGI", "AI bias", "AI fairness", "transformers", "neural networks", "supervised learning", "unsupervised learning",
        "reinforcement learning", "AI in gaming", "AI in art", "AI in music", "AI regulation", "AI safety", "AI in law",
        "sentiment analysis", "speech recognition", "AI-driven recommendation systems", "language models", "LLMs",
        "prompt engineering", "zero-shot learning", "few-shot learning", "GPT models", "BERT", "open-source AI", "AI in agriculture",
        "AI vs human intelligence", "AI hallucination", "AI training data", "AI datasets", "self-supervised learning", "AI in logistics",
        "chatbots", "AI-powered search engines", "AI startups", "AI in space exploration", "AI-powered translation",
        "multimodal AI", "AI and privacy", "AI and surveillance", "AI in warfare", "ethical dilemmas in AI", "AI in marketing",
        "AI-generated content", "deepfakes", "AI and job automation", "AI and creativity", "AI in smart homes", "AI in customer service",
        "explainable AI", "AI acceleration hardware", "edge AI", "AI in mobile apps", "AI chipsets", "openAI", "Anthropic",
        "Mistral AI", "Meta’s LLaMA", "Google DeepMind", "Gemini models", "Groq chips", "AI benchmarks", "AI evaluations",
        "RAG systems", "vector databases for AI", "FAISS", "LangChain", "Autogen", "AutoGPT", "agentic AI", "tool-using AI",
        "embodied AI", "LLM fine-tuning", "RLHF", "tokenization in AI", "attention mechanism", "cross-attention vs self-attention",
        "multi-head attention", "diffusion models", "AI in video generation", "Sora", "AI for climate science", "AI for social good",
        "AI coding assistants", "AI in neuroscience", "AI-based drug discovery", "AI for mental health", "AI in architecture"
    ])
]

LLM_EVAL_PROMPT = """
You are an expert evaluator. Assess the quality of the assistant's answer to the user's question based on retrieved documents and images.

Provide a score from 1 (poor) to 5 (excellent) for each of the following criteria:

1. Faithfulness: Is the answer factually consistent with the retrieved context?
2. Relevance: Does the answer directly answer the question?
3. Groundedness: Is the answer based on the retrieved content and not hallucinated?
4. Helpfulness: Is the answer clear, useful, and complete?

Use this format:
Faithfulness: X
Relevance: X
Groundedness: X
Helpfulness: X

Answer:

User Question:
{question}

Answer from Assistant:
{answer}

Retrieved Text Context:
{retrieved_text}

Image Descriptions:
{image_descriptions}
"""

LLM_EVAL_PROMPT_ANSWER = """
You are an expert evaluator. Assess the quality of the assistant's answer to the user's question based on retrieved documents and images.

Provide a score 0 or 1: If model have good answer return 1 else 0

Base on:
Faithfulness
Relevance
Groundedness
Helpfulness

Your output should be only 1 or 0. No word only 1 or 1

Answer:

User Question:
{question}

Answer from Assistant:
{answer}

Retrieved Text Context:
{retrieved_text}

Image Descriptions:
{image_descriptions}
"""
