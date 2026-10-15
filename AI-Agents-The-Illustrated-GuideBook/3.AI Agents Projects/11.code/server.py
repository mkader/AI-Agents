import sys
from pathlib import Path
from typing import Optional

# Direct script execution needs the parent directory for VibeStudio imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP
import uvicorn
from VibeStudio.main import run_doc_flow

# create FastMCP instance
mcp = FastMCP("doc-writer")

@mcp.tool()
def write_documentation(repo_url: str) -> str:
    """
    Generates the documentation for the given GitHub repository URL.
    Notify the user when the generation is complete.

    Args:
        repo_url (str): The URL of the GitHub repo.
    
    Returns:
        str: Returns a message when the documentation is completed successfully.
    """
    try:
        if not repo_url.startswith(("http://", "https://")):
            raise ValueError("Invalid repository URL format")
        
        run_doc_flow(repo_url)
        return f"Successfully generated documentation for repo {repo_url}"
    
    except Exception as e:
        return f"Failed to generate documentation for repo {repo_url} due to {e}"

@mcp.tool()
def list_docs() -> str:
    """
    Lists the documentation files that are generated successfully.

    Args: None
    
    Returns: 
        str: Returns a nicely formatted string of generated doc files.
    """
    docs_dir = Path("docs")
    if not docs_dir.exists():
        raise ValueError("No documentation files found")
    
    output_lines = ["Generated documentation files:"]
    for doc_file in docs_dir.glob("*.mdx"):
        output_lines.append(f"- docs/{doc_file.name}")
    return "\n".join(output_lines)

@mcp.tool()
def view_content(file_path: str) -> str:
    """
    Displays the content of a generated documentation file.
    
    Args:
        file_path (str): Relative path to the documentation file (e.g., 'docs/overview.mdx')
    
    Returns:
        str: Content of the file or error message.
    """
    try:
        if not file_path.startswith("docs/") or "../" in file_path:
            raise ValueError("Invalid file path")
            
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        if not path.is_file():
            raise ValueError(f"Path {file_path} is not a file")
        if path.suffix not in {".mdx", ".md"}:
            raise ValueError("Only documentation files (.mdx/.md) can be viewed")
            
        return path.read_text(encoding="utf-8")
        
    except Exception as e:
        return f"Failed to view documentation: {str(e)}"

# Bound shutdown waiting because MCP clients can keep SSE connections open.
if __name__ == "__main__":
    uvicorn.run(
        mcp.sse_app(),
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
        timeout_graceful_shutdown=5,
    )