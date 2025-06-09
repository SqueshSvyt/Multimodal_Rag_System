import base64

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os
from typing import Optional, List, Dict, Any
import logging


class GeminiAgent:
    """
    A class to handle question answering using Google's Gemini model via LangChain.
    """

    def __init__(
            self,
            model_name: str = "gemini-1.5-pro",
            temperature: float = 0.7,
            max_tokens: Optional[int] = 1000,
            system_prompt: Optional[str] = None
    ):
        """
        Initialize the Gemini Question Answerer.

        Args:
            api_key: Google API key. If None, will look for GOOGLE_API_KEY env variable
            model_name: Gemini model to use (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash')
            temperature: Controls randomness in responses (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response
            system_prompt: Default system prompt for the model
        """
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "Google API key must be provided either as parameter or GOOGLE_API_KEY environment variable")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
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
        Answer a single question using the Gemini model.

        Args:
            question: The user's question
            context: context to help answer the question
            custom_system_prompt: Optional custom system prompt for this specific question

        Returns:
            Dictionary containing the answer and metadata
            :param custom_system_prompt:
            :param question:
            :param context:
            :param images:
        """
        try:
            system_prompt = custom_system_prompt or self.system_prompt
            messages = []

            if system_prompt:
                messages.append(HumanMessage(content=f"System: {system_prompt}"))

            if context:
                messages.append(HumanMessage(content=f"Articles: {context}"))

            if images:
                messages.append(HumanMessage(content=f"Images: {images}"))

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
        Answer a single question using the Gemini model.

        Args:
            query: The user's question
            custom_system_prompt: Optional custom system prompt for this specific question

        Returns:
            Dictionary containing the answer and metadata
            :param custom_system_prompt:
            :param query:
        """
        try:
            system_prompt = custom_system_prompt or self.system_prompt
            messages = []

            if system_prompt:
                messages.append(HumanMessage(content=f"System: {system_prompt}"))

            messages.append(HumanMessage(content=f"{query}"))

            self.logger.info(f"Processing quary: {query[:100]}...")
            response = self.llm.invoke(messages)

            return {
                "query": query,
                "answer": response.content,
                "model": self.model_name,
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
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
            custom_prompt: Optional[str] = "Describe this image in detail. Describe the texts and patterns in image",
            image_type: Optional[str] = '.jpg'
    ) -> Dict[str, Any]:
        """
        Use Gemini to describe an image.

        Args:
            image_path: Path to the image file
            custom_prompt: Optional prompt to guide the image description

        Returns:
            Dictionary containing the description and metadata
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read and encode the image as base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            messages = [
                HumanMessage(content=f"System: You are a helpful assistant that describes images."),
                HumanMessage(content=f"Image (base64), {image_type}: {encoded_image}"),
                HumanMessage(content=custom_prompt)
            ]

            self.logger.info(f"Describing image at: {image_path}")
            response = self.llm.invoke(messages)

            return {
                "image_path": image_path,
                "description": response.content,
                "model": self.model_name,
                "success": True,
                "error": None
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

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            convert_system_message_to_human=True
        )

        self.logger.info("Model settings updated successfully")