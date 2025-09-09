"""Image validation and selection utilities.

This module provides functionality for validating document images,
extracting metadata, and displaying image information.
"""

from pathlib import Path

from PIL import Image
from rich import print as rprint
from rich.console import Console
from rich.table import Table


class ImageValidator:
    """Handles validation and information display for document images."""

    def __init__(self):
        """Initialize the image validator."""
        self.console = Console()

    def validate_and_display_image(self, image_path: str) -> str:
        """Validate image and display information.

        Args:
            image_path: Path to the image file

        Returns:
            Validated image path or None if invalid
        """
        rprint("[bold blue]🎯 Using document image selection...[/bold blue]")

        if not image_path:
            rprint("[red]❌ No document image specified[/red]")
            return None

        # Validate selected image exists
        path_obj = Path(image_path)
        if path_obj.exists():
            rprint(f"[bold green]🎉 Document ready: {path_obj.name}[/bold green]")

            # Display image info
            try:
                self._display_image_info(path_obj)
            except Exception as e:
                rprint(f"[yellow]⚠️ Could not read image info: {e}[/yellow]")

            self.console.rule("[bold blue]Document Selection Complete[/bold blue]")
            return str(path_obj)
        else:
            rprint(f"[red]❌ Image file does not exist: {path_obj}[/red]")
            rprint("[yellow]💡 Update TEST_IMAGE in configuration cell[/yellow]")
            self.console.rule("[bold blue]Document Selection Complete[/bold blue]")
            return None

    def _display_image_info(self, image_path: Path):
        """Display detailed image information in a table.

        Args:
            image_path: Path object for the image
        """
        with Image.open(image_path) as img:
            width, height = img.size
            format_info = img.format or "Unknown"

            image_info_table = Table(
                title="📊 Document Image Information", border_style="green"
            )
            image_info_table.add_column("Property", style="cyan")
            image_info_table.add_column("Value", style="yellow")

            image_info_table.add_row("Filename", image_path.name)
            image_info_table.add_row("Format", format_info)
            image_info_table.add_row("Dimensions", f"{width} × {height} pixels")
            image_info_table.add_row(
                "File Size", f"{image_path.stat().st_size / 1024:.1f} KB"
            )
            image_info_table.add_row("Full Path", str(image_path))
            image_info_table.add_row("Document Type", "To be detected...")

            self.console.print(image_info_table)

    def get_image_metadata(self, image_path: str) -> dict:
        """Get image metadata without display.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing image metadata
        """
        if not image_path or not Path(image_path).exists():
            return {}

        try:
            path_obj = Path(image_path)
            with Image.open(path_obj) as img:
                width, height = img.size
                format_info = img.format or "Unknown"

                return {
                    "filename": path_obj.name,
                    "format": format_info,
                    "width": width,
                    "height": height,
                    "file_size_kb": path_obj.stat().st_size / 1024,
                    "full_path": str(path_obj),
                    "exists": True,
                }
        except Exception:
            return {"exists": False, "error": "Could not read image"}

    def check_image_compatibility(self, image_path: str) -> tuple[bool, str]:
        """Check if image is compatible for processing.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_compatible, message)
        """
        if not image_path:
            return False, "No image path provided"

        path_obj = Path(image_path)
        if not path_obj.exists():
            return False, f"Image file does not exist: {image_path}"

        try:
            with Image.open(path_obj) as img:
                width, height = img.size
                format_info = img.format

                # Check format compatibility
                if format_info not in ["PNG", "JPEG", "JPG", "WEBP"]:
                    return (
                        False,
                        f"Unsupported format: {format_info}. Use PNG, JPEG, or WEBP.",
                    )

                # Check dimensions
                if width < 100 or height < 100:
                    return (
                        False,
                        f"Image too small: {width}x{height}. Minimum 100x100 pixels.",
                    )

                if width > 5000 or height > 5000:
                    return (
                        False,
                        f"Image too large: {width}x{height}. Maximum 5000x5000 pixels.",
                    )

                # Check file size
                file_size_mb = path_obj.stat().st_size / (1024 * 1024)
                if file_size_mb > 20:
                    return False, f"File too large: {file_size_mb:.1f}MB. Maximum 20MB."

                return True, f"Image compatible: {width}x{height}, {format_info}"

        except Exception as e:
            return False, f"Error reading image: {e}"


def validate_document_image(image_path: str) -> str:
    """Convenience function to validate a document image.

    Args:
        image_path: Path to the image file

    Returns:
        Validated image path or None if invalid
    """
    validator = ImageValidator()
    return validator.validate_and_display_image(image_path)


def check_image_requirements(image_path: str) -> bool:
    """Quick check if image meets processing requirements.

    Args:
        image_path: Path to the image file

    Returns:
        True if image is compatible, False otherwise
    """
    validator = ImageValidator()
    is_compatible, message = validator.check_image_compatibility(image_path)

    if is_compatible:
        rprint(f"[green]✅ {message}[/green]")
    else:
        rprint(f"[red]❌ {message}[/red]")

    return is_compatible
