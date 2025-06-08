import gradio as gr
from dotenv import load_dotenv

from src.prompts.system_prompt import system_prompt_gemini_answer
from src.ui.css.styles import Styles
from src.utils.gemini_agent import GeminiQuestionAnswerer
from src.utils.query_engine import QueryEngine

load_dotenv()

text_engine = QueryEngine(
    index_path="data/faiss/articles_text.index",
    metadata_path="data/metadata/metadata_article.pkl",
    data_type="text"
)

image_engine = QueryEngine(
    index_path="data/faiss/image.index",
    metadata_path="data/metadata/metadata_image.pkl",
    data_type="image"
)

qa = GeminiQuestionAnswerer(
    model_name="gemini-1.5-flash",
    temperature=0.7,
    system_prompt=system_prompt_gemini_answer
)


def dummy_retrieve(query):
    result_text = text_engine.query(query, k=3)
    result_image = image_engine.query(query, k=10)

    urls = []
    images = []

    images_with_description = ""
    combined_text = ""
    threshold_images_to_answer = 5

    for result in result_text:
        combined_text = " \n".join(combined_text + result['content'])
        urls.append(result['article_url'])

    for i, image in enumerate(result_image):
        if i < threshold_images_to_answer:
            images_with_description += f"{i} with url {image['image_urls']}. {image['image_descriptions']}\n"
        images.append(image['image_urls'])

    result = qa.answer_question(query, context=combined_text, images=images_with_description)

    return "\n".join([f"- [{url}]({url})" for url in urls]), images, result["answer"]


def show_html_response(query):
    return f"<div style='color: #333; font-size: 16px;'>You asked: <strong>{query}</strong></div>"


def query_rag_system(query):
    llm_response = f"🤖 <strong>LLM Output:</strong> AI is transforming healthcare..."
    urls = [
        "<a href='https://example.com/article1' target='_blank'>Article 1</a>",
        "<a href='https://example.com/article2' target='_blank'>Article 2</a>",
    ]
    html = llm_response + "<br><br><strong>🔗 Relevant Links:</strong><ul>" + "".join(
        f"<li>{url}</li>" for url in urls) + "</ul>"
    return html


def create_interface():
    styles = Styles(css_file_path='src/ui/css/style.css')

    with gr.Blocks(title="Multimodal RAG System", css=styles.get_css()) as demo:

        gr.Markdown("# 🔍 Multimodal RAG System")
        gr.Markdown("Ask a question and retrieve related **articles** and **images**.")

        query_input = gr.Textbox(label="🧠 Enter your query", placeholder="e.g. AI in healthcare", lines=1,
                                 elem_id="input-textbox")
        retrieve_btn = gr.Button("🔎 Retrieve")

        html_output = gr.HTML()

        urls_output = gr.Markdown(label="📄 Most Relevant URLs", elem_classes="links-list")
        images_output = gr.Gallery(label="🖼️ Relevant Images", columns=3, height="auto", elem_classes=["gallery-item"])

        retrieve_btn.click(fn=dummy_retrieve, inputs=query_input, outputs=[urls_output, images_output, html_output])
        #query_input.change(fn=query_rag_system, inputs=query_input, outputs=html_output)

    return demo