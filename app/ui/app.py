import gradio as gr

def dummy_retrieve(query):
    # Dummy placeholders for demo UI only
    processed_text = f"LLM processed text for query: '{query}'"
    urls = [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3"
    ]
    images = [
        "https://dl-staging-website.ghost.io/content/images/2024/11/unnamed--23-.png",
        "https://dl-staging-website.ghost.io/content/images/2022/10/84086bd8-fb32-4342-a502-2fdcd6401767--1-.gif",
        "https://dl-staging-website.ghost.io/content/images/2024/11/unnamed--23-.png",
        "https://dl-staging-website.ghost.io/content/images/2024/11/unnamed--23-.png",
        "https://dl-staging-website.ghost.io/content/images/2024/11/unnamed--23-.png",
        "https://dl-staging-website.ghost.io/content/images/2024/11/unnamed--23-.png",
    ]
    return "\n".join([f"- [{url}]({url})" for url in urls]), images

def show_html_response(query):
    return f"<div style='color: #333; font-size: 16px;'>You asked: <strong>{query}</strong></div>"


def query_rag_system(query):
    # Simulate LLM + Retrieval
    llm_response = f"🤖 <strong>LLM Output:</strong> AI is transforming healthcare..."
    urls = [
        "<a href='https://example.com/article1' target='_blank'>Article 1</a>",
        "<a href='https://example.com/article2' target='_blank'>Article 2</a>",
    ]
    html = llm_response + "<br><br><strong>🔗 Relevant Links:</strong><ul>" + "".join(f"<li>{url}</li>" for url in urls) + "</ul>"
    return html


with gr.Blocks(css='./app/ui/style.css') as demo:
    gr.Markdown("# 🔍 Multimodal RAG System")
    gr.Markdown("Ask a question and retrieve related **articles** and **images**.")

    query_input = gr.Textbox(label="🧠 Enter your query", placeholder="e.g. AI in healthcare", lines=1, elem_id="input-textbox")
    retrieve_btn = gr.Button("🔎 Retrieve")


    html_output = gr.HTML()

    urls_output = gr.Markdown(label="📄 Most Relevant URLs", elem_classes="links-list")
    images_output = gr.Gallery(label="🖼️ Relevant Images", columns=3, height="auto", elem_classes=["gallery-item"])

    retrieve_btn.click(fn=dummy_retrieve, inputs=query_input, outputs=[urls_output, images_output])

    query_input.change(fn=query_rag_system, inputs=query_input, outputs=html_output)

demo.launch()