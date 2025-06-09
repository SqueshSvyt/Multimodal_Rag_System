from typing import List, Dict, Any, Optional
from src.agents.gemini_agent import GeminiAgent
from src.agents.groq_agent import GroqAgent
from src.utils.query_engine import QueryEngine
from src.utils.file_reader import ComprehensiveFileReader


class Controller:
    """A class to manage query processing and retrieval for the Multimodal RAG System.

    The `Controller` class coordinates interactions between the Gemini and Groq agents,
    query engines for text and images, and the file reader to process user queries and
    uploaded files, generating responses with relevant articles, images, and HTML output.

    Attributes:
        text_engine (QueryEngine): Query engine for text-based similarity search.
        image_engine (QueryEngine): Query engine for image-based similarity search.
        qa (GeminiAgent): Gemini agent for answering questions.
        translator (GeminiAgent): Gemini agent for query preprocessing.
        image_decoder (GroqAgent): Groq agent for generating image descriptions.
        file_reader (ComprehensiveFileReader): File reader for handling uploaded files.
    """

    def __init__(
            self,
            text_index_path: str = "data/faiss/articles_text.index",
            text_metadata_path: str = "data/metadata/metadata_article.pkl",
            image_index_path: str = "data/faiss/image.index",
            image_metadata_path: str = "data/metadata/metadata_image.pkl",
            qa_model: str = "gemini-1.5-flash",
            translator_model: str = "gemini-1.5-flash",
            image_decoder_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
            qa_system_prompt: Optional[str] = None,
            translator_system_prompt: Optional[str] = None
    ):
        """Initialize the Controller with query engines and agents.

        Args:
            text_index_path (str): Path to the FAISS index for text.
            text_metadata_path (str): Path to the metadata file for text.
            image_index_path (str): Path to the FAISS index for images.
            image_metadata_path (str): Path to the metadata file for images.
            qa_model (str): Model name for the QA Gemini agent.
            translator_model (str): Model name for the translator Gemini agent.
            image_decoder_model (str): Model name for the Groq image decoder agent.
            qa_system_prompt (str, optional): System prompt for the QA agent.
            translator_system_prompt (str, optional): System prompt for the translator agent.
        """
        self.text_engine = QueryEngine(
            index_path=text_index_path,
            metadata_path=text_metadata_path,
            data_type="text"
        )
        self.image_engine = QueryEngine(
            index_path=image_index_path,
            metadata_path=image_metadata_path,
            data_type="image"
        )
        self.qa = GeminiAgent(
            model_name=qa_model,
            temperature=0.7,
            system_prompt=qa_system_prompt
        )
        self.translator = GeminiAgent(
            model_name=translator_model,
            temperature=0.5,
            system_prompt=translator_system_prompt
        )
        self.image_decoder = GroqAgent(
            model_name=image_decoder_model,
            temperature=0.5
        )
        self.file_reader = ComprehensiveFileReader()

    def show_html_image_response(self, image_urls: List[str], caption: Optional[str] = None) -> str:
        """Generate HTML for displaying images with an optional caption.

        Args:
            image_urls (List[str]): List of URLs for images to display.
            caption (str, optional): Caption for the images.

        Returns:
            str: HTML string for rendering images.
        """
        caption_html = ""
        if caption:
            caption_html = f'<div class="image-caption" style="margin-top: 10px; font-style: italic; color: #7f8c8d; font-size: 0.9em;">{caption}</div>'

        images = ""
        for image_url in image_urls:
            images += f"""
                <img src="{image_url}" 
                     alt="Response Image" 
                     class="responsive-image" 
                     style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
            """

        return f"""<div class="content-container">
            <div class="image-container" style="text-align: center; margin: 25px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                {images}
                <div class="error-message" style="display: none; margin-top: 10px; color: #e74c3c; font-style: italic;">
                    Image could not be loaded
                </div>
                {caption_html}
            </div>
        </div>"""

    def show_html_response(self, query: str) -> str:
        """Generate HTML for displaying text responses.

        Args:
            query (str): The text response to display.

        Returns:
            str: HTML string for rendering the text response.
        """
        return (f"<div class=\"content-container\">"
                f"  <div style='color: #333; font-size: 16px;'>Answer: "
                f"      <div>"
                f"          {query}"
                f"      </div>"
                f"  </div>"
                f"</div>")

    def parse_output(self, output: str) -> Dict[str, Any]:
        """Parse output string into text and image URLs.

        Args:
            output (str): String with text and optional image URLs (e.g., "text|image_url|").

        Returns:
            Dict[str, Any]: Dictionary with 'text', 'image_urls', and 'error' fields.
        """
        try:
            parts = output.split("|")
            text = parts[0].strip() if parts else ""
            image_urls = [url for i, url in enumerate(parts) if i > 0 and url.startswith("https://")]
            return {"text": text, "image_urls": image_urls, "error": None}
        except Exception as e:
            return {"text": None, "image_urls": [], "error": str(e)}

    def retrieve(self, question: str, uploaded_files: Optional[List[str]] = None) -> tuple:
        """Process a query with optional uploaded files and retrieve relevant results.

        Args:
            question (str): The user's query.
            uploaded_files (List[str], optional): List of file paths for uploaded files.

        Returns:
            tuple: (Markdown string of article URLs, list of image URLs, HTML output).
        """
        # Process uploaded files
        all_file_content = []
        if uploaded_files:
            for file_path in uploaded_files:
                result = self.file_reader.read_file(file_path)
                image_types = ['.jpg', '.jpeg', '.png', '.gif']
                if result['file_type'] in image_types:
                    result['content'] = self.image_decoder.create_image_description(result['file_path'])['description']

                if result['success']:
                    file_info = f"--- Content from {result['file_name']} ({result['file_type']}) ---"
                    all_file_content.append(f"{file_info}\n{result['content'] or ''}")
                else:
                    error_info = f"--- Failed to read {result['file_name']} ---"
                    error_msg = f"Error: {result['error']}"
                    all_file_content.append(f"{error_info}\n{error_msg}")

        combined_file_content = "\n\n".join(all_file_content) if all_file_content else None
        enhanced_question = f"{question}\n\nAdditional context from uploaded file:\n{combined_file_content}" if combined_file_content else question

        # Preprocess query
        query = self.translator.simple_query(enhanced_question)['query']

        # Retrieve text and image results
        result_text = self.text_engine.query(query, k=3)
        result_image = self.image_engine.query(query, k=10)

        urls = [result['article_url'] for result in result_text]
        images = [image['image_urls'] for image in result_image]
        combined_text = "\n".join(result['content'] for result in result_text)

        threshold_images_to_answer = 5
        images_with_description = "\n".join(
            f"{i} with url {image['image_urls']}. {image['image_descriptions']}"
            for i, image in enumerate(result_image) if i < threshold_images_to_answer
        )

        # Get answer from QA agent
        result = self.qa.answer_question(query, context=combined_text, images=images_with_description)
        parsed_output = self.parse_output(result['answer'])

        # Generate HTML output
        image_html = self.show_html_image_response(parsed_output['image_urls']) if parsed_output['image_urls'] else ""
        html_output = self.show_html_response(parsed_output['text']) + image_html

        # Format URLs as Markdown
        urls_markdown = "\n".join(f"- [{url}]({url})" for url in urls)

        return urls_markdown, images, html_output

    def clear(self) -> tuple:
        """Clear input fields.

        Returns:
            tuple: Empty values for query input and file upload fields.
        """
        return "", None