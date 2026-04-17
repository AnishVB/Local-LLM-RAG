from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"

FILES = [
    ROOT / "app.py",
    ROOT / "brain.py",
    ROOT / "processor.py",
    ROOT / "requirements.txt",
    ROOT / ".env",
]


def create_structure() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    for file_path in FILES:
        if not file_path.exists():
            file_path.touch()


if __name__ == "__main__":
    create_structure()