import gradio as gr
from src.app.css.styles import Styles


class AppBuilder:
    """A class to manage the Gradio-based user interface for the Multimodal RAG System.

    The `UI` class sets up the Gradio interface, including input fields for queries and file uploads,
    buttons for actions, and output displays for articles, images, and HTML responses.

    Attributes:
        styles (Styles): The styling object for custom CSS.
        demo (gr.Blocks): The Gradio Blocks interface.
    """

    def __init__(self, css_file_path: str = 'src/app/css/style.css'):
        """Initialize the UI with custom CSS styling.

        Args:
            css_file_path (str, optional): Path to the CSS file for styling. Defaults to 'src/app/css/style.css'.
        """
        self.styles = Styles(css_file_path=css_file_path)
        self.demo = None

    def show_html(self) -> str:
        """Generate default HTML for an empty response.

        Returns:
            str: HTML string indicating no response is available.
        """
        return """
            <div class="content-container">
                <div style='color: #ecf0f1; font-size: 16px; font-style: italic;'>
                    No response available yet.
                </div>
            </div>
        """

    def create_interface(self, retrieve_fn, clear_fn) -> gr.Blocks:
        """Create the Gradio interface for the Multimodal RAG System.

        Args:
            retrieve_fn (callable): Function to handle retrieval (takes query and files as input).
            clear_fn (callable): Function to clear input fields.

        Returns:
            gr.Blocks: The configured Gradio Blocks interface.
        """
        with gr.Blocks(title="Multimodal RAG System", css=self.styles.get_css()) as demo:
            gr.Markdown("# 🔍 Multimodal RAG System")
            gr.Markdown("Ask a question and retrieve related **articles** and **images**.")

            query_input = gr.Textbox(
                label="🧠 Enter your query",
                placeholder="e.g. AI in healthcare",
                lines=1,
                elem_id="input-textbox"
            )

            file_upload = gr.File(
                label="📎 Upload File (optional)",
                file_types=[".txt", ".pdf", ".docx", ".jpg", ".jpeg", ".png", ".gif"],
                file_count="multiple"
            )

            with gr.Row():
                retrieve_btn = gr.Button("🔎 Retrieve", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            html_output = gr.HTML(value=self.show_html)

            urls_output = gr.Markdown(label="📄 Most Relevant Articles")
            images_output = gr.Gallery(
                label="🖼️ Relevant Images",
                columns=4,
                height="auto",
                object_fit="scale-down",
                elem_classes=["gallery-item"]
            )

            retrieve_btn.click(
                fn=retrieve_fn,
                inputs=[query_input, file_upload],
                outputs=[urls_output, images_output, html_output]
            )

            clear_btn.click(
                fn=clear_fn,
                outputs=[query_input, file_upload]
            )

        self.demo = demo
        return demo

    def launch(self, **kwargs):
        """Launch the Gradio interface.

        Args:
            **kwargs: Additional arguments to pass to `gr.Blocks.launch` (e.g., server_name, server_port).

        Returns:
            The result of the Gradio launch operation.
        """
        if self.demo is None:
            raise ValueError("Interface not created. Call create_interface() first.")
        return self.demo.launch(**kwargs)