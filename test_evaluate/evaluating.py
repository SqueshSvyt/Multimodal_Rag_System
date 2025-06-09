import json
import openai
import re

from src.app.controllers.app_controller import Controller
from src.prompts.system_prompt import system_prompt_gemini_answer_agent, system_prompt_gemini_query_preprocess_agent
from test_evaluate.questinons_and_prompts import LLM_EVAL_PROMPT, LLM_EVAL_PROMPT_ANSWER
from src.agents.groq_agent import GroqAgent
from dotenv import load_dotenv
from typing import Dict


load_dotenv()


def extract_query_from_html(html_response: str) -> str:
    match = re.search(r'<div[^>]*>\s*([^<]+)\s*</div>\s*</div>\s*</div>', html_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def evaluate_with_langchain(
        question: str,
        answer: str,
        retrieved_text: str,
        image_descriptions: str,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.2,
        prompt_template: str = None,
) -> Dict[str, int]:
    if prompt_template is None:
        prompt_template = LLM_EVAL_PROMPT

    prompt = prompt_template.format(
        question=question,
        answer=answer,
        retrieved_text=retrieved_text,
        image_descriptions=image_descriptions
    )

    chat = GroqAgent(model_name=model_name, temperature=temperature)

    response = chat.simple_query(query=prompt, custom_system_prompt=prompt)

    print("Evaluation response:\n", response['answer'])

    # Parse the response into metrics
    metrics = {}
    for line in response['answer'].split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            try:
                metrics[key.strip().lower()] = int(val.strip())
            except ValueError:
                continue
    return metrics


def load_json_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


controller = Controller(
    text_index_path="../data/faiss/articles_text.index",
    text_metadata_path="../data/metadata/metadata_article.pkl",
    image_index_path="../data/faiss/image.index",
    image_metadata_path="../data/metadata/metadata_image.pkl",
    qa_system_prompt=system_prompt_gemini_answer_agent,
    translator_system_prompt=system_prompt_gemini_query_preprocess_agent
)

results = load_json_file('ai_questions_answers.json')

parsed = [extract_query_from_html(answer['html_answer']) for answer in results]

evaluations = []
for i, result in enumerate(results):
    evaluation = evaluate_with_langchain(
        question=result['question'],
        answer=parsed[i],
        retrieved_text=result['article_urls'],
        image_descriptions=result['image_urls']
    )

    print("Evaluation Scores:", evaluation)

    evaluations.append(evaluation)

with open("0_5_model_answer.json", "w", encoding="utf-8") as f:
    json.dump(evaluations, f, ensure_ascii=False, indent=2)

print("Finished processing all questions. Results saved to 0_5_model_answer.json.")
