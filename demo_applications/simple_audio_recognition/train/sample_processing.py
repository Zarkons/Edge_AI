import os
import pathlib
import random

DEFAULT_LABEL_NAMES = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

def get_build_dir():
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    
    if workspace_dir:
        actual_save_dir = os.path.join(workspace_dir, "demo_applications/simple_audio_recognition/train/_build_dir/")
    else:
        # 2. Universal Fallback: Calculate path dynamically relative to this file's position
        # (This handles standard python runs, VS Code debugging, or bazel tests)
        current_file_path = os.path.abspath(__file__)
        # Adjust the number of parents based on your folder depth to reach the root
        repo_root = pathlib.Path(current_file_path).parents[2] 
        actual_save_dir = os.path.join(repo_root, "demo_applications/simple_audio_recognition/train/_build_dir/")
        
    # Ensure the directory actually exists on your Mac before saving files to it
    os.makedirs(actual_save_dir, exist_ok=True)
    return pathlib.Path(actual_save_dir)

def extract_command_from_filename(wav_path: str) -> str:
    """Extracts the command label from the filename of a .wav file."""
    return pathlib.Path(wav_path).parent.name

def load_random_audio_sample(build_dir: pathlib.Path) -> str:
    """Returns the path to a random .wav file from the mini_speech_commands dataset."""
    data_dir = build_dir / "data/mini_speech_commands_extracted/mini_speech_commands"
    all_wav_files = list(data_dir.rglob("*.wav"))
    if not all_wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}. Ensure the dataset is extracted.")
    wav_file = random.choice(all_wav_files)
    return str(wav_file)

def save_label_names(label_names: list[str]) -> pathlib.Path:
    """Persist the training label order for inference."""
    label_file = get_build_dir() / "label_names.txt"
    label_file.write_text("\n".join(label_names) + "\n", encoding="utf-8")
    return label_file

def get_label_names() -> list:
    """Return the trained label names, falling back to the default order."""
    label_file = get_build_dir() / "label_names.txt"
    if label_file.exists():
        labels = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines()]
        return [label for label in labels if label]
    return DEFAULT_LABEL_NAMES