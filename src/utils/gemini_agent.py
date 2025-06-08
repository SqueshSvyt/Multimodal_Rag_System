from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os
from typing import Optional, List, Dict, Any
import logging


class GeminiQuestionAnswerer:
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
            context: Optional context to help answer the question
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

    def chat_conversation(
            self,
            messages_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Handle a conversation with message history.

        Args:
            messages_history: List of dicts with 'role' and 'content' keys
                            Role can be 'user' or 'assistant'

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Convert message history to LangChain format
            langchain_messages = []

            # Add system prompt
            if self.system_prompt:
                langchain_messages.append(HumanMessage(content=f"System: {self.system_prompt}"))

            # Add conversation history
            for msg in messages_history:
                if msg['role'] == 'user':
                    langchain_messages.append(HumanMessage(content=msg['content']))

            response = self.llm.invoke(langchain_messages)

            return {
                "response": response.content,
                "model": self.model_name,
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Error in conversation: {str(e)}")
            return {
                "response": None,
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