"""Template loading and rendering for LLM prompts."""

from importlib.resources import files
from jinja2 import Environment, BaseLoader, TemplateNotFound


class ResourceLoader(BaseLoader):
    """Jinja2 loader that loads templates from package resources."""
    
    def __init__(self, package_path):
        """
        Initialize loader.
        
        Args:
            package_path: Traversable path to the templates directory
        """
        self.package_path = package_path
    
    def get_source(self, environment, template):
        """Load template from package resources."""
        try:
            template_file = self.package_path / template
            if not template_file.is_file():
                raise TemplateNotFound(template)
            
            source = template_file.read_text(encoding='utf-8')
            # Return (source, filename, uptodate_function)
            # uptodate_function returns True to indicate template is always fresh
            return source, None, lambda: True
        except Exception as e:
            raise TemplateNotFound(template) from e


class PromptRenderer:
    """Renders LLM prompts from Jinja2 templates."""
    
    def __init__(self):
        """Initialize renderer with templates from package resources."""
        prompts_path = files('papervibe').joinpath('prompts')
        loader = ResourceLoader(prompts_path)
        self.env = Environment(loader=loader, autoescape=False)
    
    def render_rewrite_abstract_system(self) -> str:
        """Render system prompt for abstract rewriting."""
        template = self.env.get_template('rewrite_abstract_system.j2')
        return template.render()
    
    def render_rewrite_abstract_user(self, original_abstract: str) -> str:
        """
        Render user prompt for abstract rewriting.
        
        Args:
            original_abstract: The original abstract text
            
        Returns:
            Rendered user prompt
        """
        template = self.env.get_template('rewrite_abstract_user.j2')
        return template.render(original_abstract=original_abstract)
    
    def render_gray_out_system(self) -> str:
        """Render system prompt for graying out sentences."""
        template = self.env.get_template('gray_out_system.j2')
        return template.render()
    
    def render_gray_out_user(self, chunk: str, gray_ratio: float) -> str:
        """
        Render user prompt for graying out sentences.

        Args:
            chunk: The text chunk to process
            gray_ratio: Target ratio of text to gray out (0.0 to 1.0)

        Returns:
            Rendered user prompt
        """
        template = self.env.get_template('gray_out_user.j2')
        return template.render(
            chunk=chunk,
            gray_ratio_percent=gray_ratio * 100,
        )

    def render_highlight_system(self) -> str:
        """Render system prompt for highlighting content."""
        template = self.env.get_template('highlight_system.j2')
        return template.render()

    def render_highlight_user(self, chunk: str, highlight_ratio: float) -> str:
        """
        Render user prompt for highlighting content.

        Args:
            chunk: The text chunk to process
            highlight_ratio: Target ratio of content to highlight (0.0 to 1.0)

        Returns:
            Rendered user prompt
        """
        template = self.env.get_template('highlight_user.j2')
        return template.render(
            chunk=chunk,
            highlight_ratio_percent=highlight_ratio * 100,
        )


# Global singleton instance
_renderer = None


def get_renderer() -> PromptRenderer:
    """Get or create the global prompt renderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = PromptRenderer()
    return _renderer
