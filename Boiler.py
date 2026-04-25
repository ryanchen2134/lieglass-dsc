from pathlib import Path
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_python_file_content(relative_path: str) -> str:
    """
    Reads the content of a Python file given its relative path 
    to the current working directory.
    """
    # Create a path object relative to the current working directory
    file_path = Path.cwd() / relative_path
    
    try:
        # read_text() handles opening and closing the file automatically
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return f"Error: The file at {relative_path} was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Example usage:
# content = get_python_file_content("scripts/my_script.py")
# print(content)

files = [
    #"deception_detection/train.py",
    "deception_detection/config.py",
    "deception_detection/models/fusion_model.py",
    "deception_detection/models/visual_model.py",
    "deception_detection/models/audio_model.py",
    "deception_detection/data/dataset.py"
]

preproc = [
    "deception_detection/data/preprocessing/extract_frames.py",
    "deception_detection/data/preprocessing/preprocess_fullframe.py.py",
    "deception_detection/data/preprocessing/preprocess_resized.py",
    "deception_detection/data/preprocessing/run_all.py",
    "deception_detection/data/preprocessing/video_prep.py",
]

mainstr = ""
for file in files:
    content = get_python_file_content(file)
    mainstr += f"\n\n--------------------\n{file}\n--------------------\n\n{content}\n\n"

with open('code.txt', 'w') as file:
    file.write(mainstr)

