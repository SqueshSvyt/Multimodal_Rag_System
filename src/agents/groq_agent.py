import base64
import os
from typing import Optional, Dict, Any
import logging
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage


class GroqAgent:
    """
    A class to handle question answering using Groq's LLaMA model via LangChain.
    """

    def __init__(
            self,
            model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: float = 0.7,
            max_tokens: Optional[int] = 1000,
            system_prompt: Optional[str] = None
    ):
        """
        Initialize the Groq Question Answerer.

        Args:
            model_name: Groq model to use (e.g., 'llama3-8b-8192', 'llama3-70b-8192')
            temperature: Controls randomness in responses (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response
            system_prompt: Default system prompt for the model
        """
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "Groq API key must be provided as GROQ_API_KEY environment variable. "
                "Please visit https://console.groq.com/docs/api-keys for details."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def answer_question(
            self,
            question: str,
            context: Optional[str] = None,
            images: Optional[str] = None,
            custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a single question using the Groq LLaMA model.

        Args:
            question: The user's question
            context: Context to help answer the question
            images: Base64 encoded image data or image description (note: LLaMA may not support images)
            custom_system_prompt: Optional custom system prompt for this specific question

        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            system_prompt = custom_system_prompt or self.system_prompt
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            if context:
                messages.append(HumanMessage(content=f"Context: {context}"))

            if images:
                self.logger.warning(
                    "LLaMA models on Groq typically do not support image processing. "
                    "Treating 'images' as a text description."
                )
                messages.append(HumanMessage(content=f"Image description: {images}"))

            messages.append(HumanMessage(content=f"Question: {question}"))

            self.logger.info(f"Processing question: {question[:100]}...")
            response = self.llm.invoke(messages)

            return {
                "question": question,
                "answer": response.content,
                "model": self.model_name,
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": None,
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }

    def simple_query(
            self,
            query: str,
            custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a single query using the Groq LLaMA model.

        Args:
            query: The user's query
            custom_system_prompt: Optional custom system prompt for this specific query

        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            system_prompt = custom_system_prompt or self.system_prompt
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=query))

            self.logger.info(f"Processing query: {query[:100]}...")
            response = self.llm.invoke(messages)

            return {
                "query": query,
                "answer": response.content,
                "model": self.model_name,
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Error answering query: {str(e)}")
            return {
                "query": query,
                "answer": None,
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }

    def create_image_description(
            self,
            image_path: str,
            custom_prompt: Optional[str] = "Describe this image in detail. Describe the texts and patterns in image. Write 250 tokens",
            image_type: Optional[str] = '.jpg'
    ) -> Dict[str, Any]:
        """
        Use Groq LLaMA to describe an image (note: LLaMA models typically do not support image processing).

        Args:
            image_path: Path to the image file
            custom_prompt: Optional prompt to guide the image description
            image_type: File extension of the image (e.g., '.jpg', '.png')

        Returns:
            Dictionary containing the description and metadata
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            IMAGE_DATA_URL = f"data:image/{image_type[1:]};base64,{encoded_image}"

            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": custom_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": IMAGE_DATA_URL
                        }
                    }
                ],
            }

            response = self.llm.invoke([message])

            return {
                "image_path": image_path,
                "description": response.content,
                "model": self.model_name,
                "success": False,
                "error": "Image processing is not supported by LLaMA models on Groq."
            }

        except Exception as e:
            self.logger.error(f"Error describing image: {str(e)}")
            return {
                "image_path": image_path,
                "description": None,
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }

    def update_settings(
            self,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            system_prompt: Optional[str] = None
    ):
        """
        Update model settings.

        Args:
            temperature: New temperature setting
            max_tokens: New max tokens setting
            system_prompt: New system prompt
        """
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if system_prompt is not None:
            self.system_prompt = system_prompt

        self.llm = ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        self.logger.info("Model settings updated successfully")