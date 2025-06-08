import os


class Styles:
    def __init__(self, css_file_path):
        """
        Initialize the Styles class with a CSS file path.

        Args:
            css_file_path (str): Path to the CSS file (e.g., "static/style.css")

        Raises:
            FileNotFoundError: If the CSS file doesn't exist.
            IOError: If there's an error reading the file.
        """
        self.css_file_path = css_file_path
        self.css_content = self._load_css()

    def _load_css(self):
        """
        Load the CSS content from the file.

        Returns:
            str: The content of the CSS file.

        Raises:
            FileNotFoundError: If the file is not found.
            IOError: If there's an error reading the file.
        """
        if not os.path.exists(self.css_file_path):
            raise FileNotFoundError(f"CSS file not found at: {self.css_file_path}")

        try:
            with open(self.css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Error reading CSS file {self.css_file_path}: {e}")

    def get_css(self):
        """
        Get the CSS content as a string.

        Returns:
            str: The raw CSS content.
        """
        return self.css_content

    def get_html_link(self, alias="/static/style.css"):
        """
        Get an HTML <link> tag for the CSS file.

        Args:
            alias (str): The URL alias for the CSS file (default: "/static/style.css").

        Returns:
            str: An HTML <link> tag.
        """
        return f'<link rel="stylesheet" type="text/css" href="{alias}">'

    def get_embedded_style(self):
        """
        Get the CSS content wrapped in a <style> tag for embedding.

        Returns:
            str: The CSS content wrapped in <style> tags.
        """
        return f"<style>{self.css_content}</style>"