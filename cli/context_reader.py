import os
from pathlib import Path

def is_binary(file_path: Path) -> bool:
    """
    Check if a file is binary by searching for null bytes in the first 1024 bytes.
    This prevents the system from trying to read images, executables, or ZIPs as text.
    """
    try:
        # Open the file in read-binary mode ('rb')
        with open(file_path, 'rb') as f:
            # Read the first 1024 bytes (enough to check for binary headers/nulls)
            chunk = f.read(1024)
            # Standard heuristic: if a null byte (0x00) exists, it's highly likely binary
            return b'\x00' in chunk
    except Exception:
        # If any reading error occurs (e.g. permission denied), treat as binary to be safe
        return True

def should_ignore(path: Path, root_dir: Path) -> bool:
    """
    Determine if a file or directory should be ignored based on common patterns.
    We exclude bulky, binary, or configuration directories to optimize performance.
    """
    # Set of directory/file names to ignore completely during traversal
    ignored_names = {
        '.git', '.github', '.vscode', 'node_modules', '__pycache__', 
        'venv', '.venv', 'env', '.env', 'dist', 'build', 'eggs', 
        'target', '.gemini', '.idea', 'results', 'output', 'models'
    }
    
    # Set of file extensions that represent binaries, assets, or database files
    ignored_extensions = {
        '.pyc', '.pyo', '.pyd', '.db', '.sqlite', '.sqlite3', 
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', 
        '.tar', '.gz', '.7z', '.rar', '.exe', '.dll', '.so', '.bin',
        '.woff', '.woff2', '.eot', '.ttf', '.mp4', '.mp3', '.wav'
    }
    
    try:
        # Calculate the path relative to the root directory
        relative = path.relative_to(root_dir)
        # Check if any parent folder in the relative path is in our ignore list
        for part in relative.parts:
            if part in ignored_names:
                return True
    except ValueError:
        # Fallback if path is outside root_dir (should not happen normally)
        pass
        
    # Check if the specific file/folder name or extension is in the ignore lists
    if path.name in ignored_names or path.suffix.lower() in ignored_extensions:
        return True
        
    return False

def read_directory_context(dir_path: str) -> str:
    """
    Recursively read all text files in a directory and format them as Markdown.
    Includes safeguards (file size and total context limit) to avoid token overflow.
    """
    # Resolve the path to an absolute path for consistency
    root = Path(dir_path).resolve()
    
    # Return error message if the path doesn't exist on disk
    if not root.exists():
        return f"[Error: Path {dir_path} does not exist]"
        
    # If the user passed a single file instead of a folder, process it directly
    if root.is_file():
        # Skip if it is binary
        if is_binary(root):
            return f"[File {root.name} is binary and was skipped]"
        try:
            # Read file content as text
            content = root.read_text(encoding='utf-8', errors='replace')
            # Return wrapped in a markdown header and codeblock
            return f"### File: {root.name}\n```\n{content}\n```\n"
        except Exception as e:
            # Catch and return any file read errors
            return f"[Error reading file {root.name}: {e}]"

    # List to store the markdown-formatted file contents
    context_parts = []
    # Track total character/byte size read so far
    total_size = 0
    # Safeguard: do not load more than 1MB of text context to prevent LLM context window crashes
    max_total_size = 1024 * 1024  
    
    # Walk the directory tree recursively
    for root_dir, dirs, files in os.walk(root):
        # Modify dirs in-place to prune and ignore unwanted directories (prevents traversing them)
        dirs[:] = [d for d in dirs if not should_ignore(Path(root_dir) / d, root)]
        
        for file in files:
            # Construct the absolute path of the file
            file_path = Path(root_dir) / file
            # Skip if file matches our ignore rules
            if should_ignore(file_path, root):
                continue
            # Skip binary files
            if is_binary(file_path):
                continue
                
            try:
                # Get the size of the file in bytes
                file_size = file_path.stat().st_size
                # Skip files larger than 200KB to prevent single large files from eating the whole context
                if file_size > 200 * 1024:  
                    continue
                    
                # If reading this file exceeds our 1MB limit, stop and add warning
                if total_size + file_size > max_total_size:
                    context_parts.append(f"\n[Warning: Directory context size exceeded limit. Remaining files skipped.]\n")
                    break
                    
                # Read text (replace decoding errors gracefully instead of crashing)
                content = file_path.read_text(encoding='utf-8', errors='replace')
                # Calculate relative path from root to show in context headers
                rel_path = file_path.relative_to(root)
                # Format file content in markdown code block
                context_parts.append(f"### File: {rel_path}\n```\n{content}\n```\n")
                # Increment total size
                total_size += file_size
            except Exception as e:
                # Document failures for individual files but continue processing others
                context_parts.append(f"[Error reading file {file}: {str(e)}]\n")
                
        # Exit outer directory walk loop if context limit is exceeded
        if total_size > max_total_size:
            break
            
    # Combine all markdown formatted contents into a single string
    return "\n".join(context_parts)
