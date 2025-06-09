from dotenv import load_dotenv

from src.app.controllers.app_controller import Controller
from src.app.interface import AppBuilder
from src.prompts.system_prompt import system_prompt_gemini_answer_agent, system_prompt_gemini_query_preprocess_agent

load_dotenv()

controller = Controller(
    text_index_path="data/faiss/articles_text.index",
    text_metadata_path="data/metadata/metadata_article.pkl",
    image_index_path="data/faiss/image.index",
    image_metadata_path="data/metadata/metadata_image.pkl",
    qa_system_prompt=system_prompt_gemini_answer_agent,
    translator_system_prompt=system_prompt_gemini_query_preprocess_agent
)

app = AppBuilder()

demo = app.create_interface(
    retrieve_fn=controller.retrieve,
    clear_fn=controller.clear
)

demo.launch()
