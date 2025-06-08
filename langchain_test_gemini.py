from src.utils.gemini_agent import GeminiQuestionAnswerer
from dotenv import load_dotenv

from src.utils.query_engine import QueryEngine
from src.prompts.system_prompt import system_prompt_gemini_answer

load_dotenv()

qa = GeminiQuestionAnswerer(
    model_name="gemini-1.5-flash",
    temperature=0.7,
    system_prompt=system_prompt_gemini_answer
)

question = "What is Machine Learning?"

text_engine = QueryEngine(
    index_path="data/faiss/articles_text.index",
    metadata_path="data/metadata/metadata_article.pkl",
    data_type="text"
)
texts = text_engine.query(question, k=3)

combined_text = ""
for result in texts:
    combined_text = " \n".join(combined_text + result['content'])

result = qa.answer_question(question, context=combined_text)

if result["success"]:
     print("Question:", result["question"])
     print("Answer:", result["answer"])
else:
     print("Error:", result["error"])