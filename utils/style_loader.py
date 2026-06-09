# ==============================================================================
# STYLE LOADER UTILITY
# ==============================================================================
import os

def load_css(file_name):
    """
    Reads a stylesheet file from the css directory and wraps it in a <style> block.
    """
    # Define absolute path to the CSS file
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "css", file_name)
    
    # Read file contents if file exists
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            return f"<style>{f.read()}</style>"
    return ""

def load_svg(file_name):
    """
    Reads an SVG file from the images/icons directory and returns it as a clean single-line string.
    """
    svg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "icons", file_name)
    if os.path.exists(svg_path):
        with open(svg_path, "r") as f:
            # Join lines stripping newlines and leading/trailing indentation
            return "".join(line.strip() for line in f)
    return ""

def clean_html(html_str):
    """
    Strips leading and trailing whitespace from each line of the HTML string
    to prevent markdown parsers from treating indented lines as code blocks.
    """
    if not html_str:
        return ""
    return "\n".join(line.strip() for line in html_str.split("\n"))


