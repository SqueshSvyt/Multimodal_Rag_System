import json

from dotenv import load_dotenv

from questinons_and_prompts import questions
from src.app.controllers.app_controller import Controller
from src.prompts.system_prompt import system_prompt_gemini_answer_agent, system_prompt_gemini_query_preprocess_agent

load_dotenv()

controller = Controller(
    text_index_path="../data/faiss/articles_text.index",
    text_metadata_path="../data/metadata/metadata_article.pkl",
    image_index_path="../data/faiss/image.index",
    image_metadata_path="../data/metadata/metadata_image.pkl",
    qa_system_prompt=system_prompt_gemini_answer_agent,
    translator_system_prompt=system_prompt_gemini_query_preprocess_agent
)

results = []

for idx, q in enumerate(questions, 1):
    print(f"Processing Q{idx}: {q}")
    try:
        urls_markdown, images, html_output = controller.retrieve(q)
        results.append({
            "question": q,
            "article_urls": urls_markdown,
            "image_urls": images,
            "html_answer": html_output
        })
    except Exception as e:
        print(f"Error on question {idx}: {e}")
        results.append({
            "question": q,
            "error": str(e)
        })


with open("ai_questions_answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Finished processing all questions. Results saved to ai_questions_answers.json.")