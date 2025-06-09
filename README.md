# Multimodal RAG System

The Multimodal Retrieval-Augmented Generation (RAG) System is a Python application that enables users to query a knowledge base, retrieve relevant articles and images, and generate answers using AI agents. The system supports text and file inputs (e.g., `.txt`, `.pdf`, `.docx`, images) and provides a user-friendly Gradio-based interface.

## Overview

The system consists of two primary components:
- **AppBuilder**: Manages the Gradio-based user interface, including input fields for queries, file uploads, buttons, and output displays for articles, images, and HTML responses.
- **Controller**: Coordinates query processing, file handling, and retrieval using Gemini and Groq AI agents, FAISS-based query engines, and a comprehensive file reader.

### Features
- Query processing with optional file uploads (`.txt`, `.pdf`, `.docx`, `.jpg`, `.jpeg`, `.png`, `.gif`).
- Retrieval of relevant articles and images using FAISS-based similarity search with embeddings from models like `all-MiniLM-L6-v2` or OpenAI.
- AI-generated answers using Gemini for question answering and query preprocessing, and Groq for image descriptions.
- Customizable UI styling via CSS.
- Support for multiple file types through `ComprehensiveFileReader`.

## Requirements

- **Python**: 3.7 or higher
- **Dependencies**: Listed in `requirements.txt`:
  ```plaintext
  # UI and App
  gradio>=3.40.0
  Pillow==10.3.0  # for image display

  # Embedding Models
  sentence-transformers==2.5.1  # for models like all-MiniLM-L6-v2
  openai>=1.84.0  # if using OpenAI embeddings

  # Vector Database
  faiss-cpu

  # Optional: Qwen and HuggingFace support
  transformers==4.41.2
  torch==2.3.0
  accelerate==0.30.1

  # Data and Utilities
  pandas==2.3.0
  numpy==1.26.4
  pdfplumber==0.11.6
  python-docx==1.1.2
  docx==0.2.4
  tqdm==4.66.4

  # Optional dependencies for LangChain integrations
  langchain==0.3.25
  langchain-openai==0.3.21
  langchain-google-genai==2.1.5
  langchain-groq==0.3.2
  pydantic>=2.8.2

  # Other utilities
  anyio==4.3.0
  requests==2.32.3
  python-dotenv~=1.1.0
  ```

- **API Access**: Credentials for Gemini (Google GenAI) and Groq APIs.
- **FAISS Indexes**: Prebuilt FAISS indexes and metadata files for text and images:
  - Text: `data/faiss/articles_text.index`, `data/metadata/metadata_article.pkl`
  - Images: `data/faiss/image.index`, `data/metadata/metadata_image.pkl`
- **CSS File**: For UI styling (default: `src/app/css/style.css`).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Credentials**:
   - Create a `.env` file in the project root:
     ```plaintext
     GOOGLE_API_KEY=<your-gemini-api-key>
     GROQ_API_KEY=<your-groq-api-key>
     OPENAI_API_KEY=<your-openai-api-key>  # If using OpenAI embeddings
     ```
   - Load environment variables using `python-dotenv`.

5. **Prepare FAISS Indexes**:
   - Ensure FAISS indexes and metadata files are available at the specified paths or generate them using your embedding model (e.g., `sentence-transformers` or OpenAI).


6. **Configure API Credentials**:
   - Start app using command:
     ```plaintext
      python startup.py
     ```
   - Load environment variables using `python-dotenv`.

## Usage

### AppBuilder (UI)

The `AppBuilder` class creates and launches the Gradio interface for user interaction.

#### Example Code

```python
from src.app.app_builder import AppBuilder
from src.app.controller import Controller

# Initialize controller
controller = Controller(
    text_index_path="data/faiss/articles_text.index",
    text_metadata_path="data/metadata/metadata_article.pkl",
    image_index_path="data/faiss/image.index",
    image_metadata_path="data/metadata/metadata_image.pkl",
    qa_model="gemini-1.5-flash",
    translator_model="gemini-1.5-flash",
    image_decoder_model="meta-llama/llama-4-scout-17b-16e-instruct"
)

# Initialize UI
app = AppBuilder(css_file_path="src/app/css/style.css")
interface = app.create_interface(
    retrieve_fn=controller.retrieve,
    clear_fn=controller.clear
)

# Launch the interface
interface.launch(server_name="0.0.0.0", server_port=7860)
```

#### Key Methods

- **`__init__(css_file_path: str)`**:
  - Initializes the UI with custom CSS.
  - Args:
    - `css_file_path` (str): Path to CSS file (default: `src/app/css/style.css`).

- **`create_interface(retrieve_fn, clear_fn) -> gr.Blocks`**:
  - Creates the Gradio interface with query input, file upload, buttons, and output displays.
  - Args:
    - `retrieve_fn` (callable): Function for query retrieval (e.g., `Controller.retrieve`).
    - `clear_fn` (callable): Function to clear inputs (e.g., `Controller.clear`).
  - Returns: A `gr.Blocks` object.

- **`launch(**kwargs)`**:
  - Launches the Gradio interface.
  - Args: Keyword arguments for `gr.Blocks.launch` (e.g., `server_name`, `server_port`).
  - Raises: `ValueError` if the interface is not created.

- **`show_html() -> str`**:
  - Returns default HTML for an empty response.

#### Interface Components
- **Query Input**: Textbox for queries (e.g., "AI in healthcare").
- **File Upload**: Supports multiple files (`.txt`, `.pdf`, `.docx`, `.jpg`, `.jpeg`, `.png`, `.gif`).
- **Buttons**:
  - **Retrieve**: Triggers query processing.
  - **Clear**: Resets inputs.
- **Outputs**:
  - **HTML Output**: Displays AI-generated answers and images.
  - **URLs Output**: Markdown list of article URLs.
  - **Images Output**: Gallery of relevant images.

### Controller

The `Controller` class manages query processing, file handling, and retrieval.

#### Example Code

```python
from src.app.controller import Controller

# Initialize controller
controller = Controller(
    text_index_path="data/faiss/articles_text.index",
    text_metadata_path="data/metadata/metadata_article.pkl",
    image_index_path="data/faiss/image.index",
    image_metadata_path="data/metadata/metadata_image.pkl",
    qa_system_prompt="QA Prompt",
    translator_system_prompt="Query Prompt"
)

# Process a query with a file
urls, images, html = controller.retrieve(
    question="What is AI in healthcare?",
    uploaded_files=["healthcare_article.pdf"]
)
print(urls, images, html)

# Clear inputs
query, files = controller.clear()
print(query, files)
```

#### Key Methods

- **`__init__(text_index_path, text_metadata_path, image_index_path, image_metadata_path, qa_model, translator_model, image_decoder_model, qa_system_prompt, translator_system_prompt)`**:
  - Initializes query engines, AI agents, and file reader.
  - Args:
    - `text_index_path` (str): Path to text FAISS index (default: `data/faiss/articles_text.index`).
    - `text_metadata_path` (str): Path to text metadata (default: `data/metadata/metadata_article.pkl`).
    - `image_index_path` (str): Path to image FAISS index (default: `data/faiss/image.index`).
    - `image_metadata_path` (str): Path to image metadata (default: `data/metadata/metadata_image.pkl`).
    - `qa_model` (str): Gemini QA model (default: `gemini-1.5-flash`).
    - `translator_model` (str): Gemini translator model (default: `gemini-1.5-flash`).
    - `image_decoder_model` (str): Groq image decoder model (default: `llama3-8b-8192`).
    - `qa_system_prompt` (Optional[str]): System prompt for QA agent.
    - `translator_system_prompt` (Optional[str]): System prompt for translator agent.

- **`retrieve(question: str, uploaded_files: Optional[List[str]]) -> tuple`**:
  - Processes a query with optional files and retrieves results.
  - Args:
    - `question` (str): User query.
    - `uploaded_files` (Optional[List[str]]): List of file paths.
  - Returns: Tuple of (Markdown URLs, image URLs, HTML output).

- **`clear() -> tuple`**:
  - Clears input fields.
  - Returns: Tuple of empty query (`""`) and file list (`None`).

- **`show_html_response(query: str) -> str`**:
  - Generates HTML for text responses.
  - Args: `query` (str): Text to display.
  - Returns: HTML string.

- **`show_html_image_response(image_urls: List[str], caption: Optional[str]) -> str`**:
  - Generates HTML for image display.
  - Args:
    - `image_urls` (List[str]): List of image URLs.
    - `caption` (Optional[str]): Image caption.
  - Returns: HTML string.

- **`parse_output(output: str) -> Dict[str, Any]`**:
  - Parses output into text and image URLs.
  - Args: `output` (str): Output string (e.g., "text|image_url|").
  - Returns: Dictionary with `text`, `image_urls`, and `error`.

#### Processing Flow
1. **File Handling**: Uses `ComprehensiveFileReader` to read files. Images are described using the Groq agent (`langchain-groq`).
2. **Query Enhancement**: Combines query with file content and preprocesses using the Gemini translator agent (`langchain-google-genai`).
3. **Retrieval**: Queries text and image FAISS indexes (`faiss-cpu`) using embeddings (`sentence-transformers` or `openai`).
4. **Answer Generation**: Uses the Gemini QA agent with retrieved context and image descriptions.
5. **Output Formatting**: Generates Markdown for URLs, a gallery for images, and HTML for answers.

## Example Output

For a query with a PDF file:
```python
# Input
question = "AI in healthcare"
uploaded_files = ["healthcare_article.pdf"]

# Output
urls_markdown = "- [https://example.com/article1](https://example.com/article1)\n- [https://example.com/article2](https://example.com/article2)"
images = ["https://example.com/image1.jpg", "https://example.com/image2.png"]
html_output = """
<div class="content-container">
  <div style='color: #333; font-size: 16px;'>Answer:
      <div>AI in healthcare improves diagnostics and treatment...</div>
  </div>
</div>
<div class="content-container">
  <div class="image-container" style="text-align: center; margin: 25px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px;">
      <img src="https://example.com/image1.jpg" alt="Response Image" class="responsive-image" style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 4px 8px;" onerror="...">
  </div>
</div>
"""
```

## Notes

- **File Types**: Supports `.txt`, `.pdf` (`pdfplumber`), `.docx` (`python-docx`), and images (`.jpg`, `.jpeg`, `.png`, `.gif`) via `Pillow`. Extend `ComprehensiveFileReader` for additional formats.
- **API Limits**: Ensure compliance with Gemini and Groq API limits for file sizes and query rates.
- **FAISS Indexes**: Generate indexes using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) or `openai` embeddings.
- **Customization**: Adjust CSS in `style.css` or modify system prompts for agent behavior.
- **Optional Dependencies**: Install `transformers`, `torch`, and `accelerate` only if using Qwen or HuggingFace models.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or report bugs.