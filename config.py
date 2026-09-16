from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATABASE_DIR = PROJECT_ROOT / "database"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSET_DIR = PROJECT_ROOT / "assets"

LITERATURE_DB_PATH = DATABASE_DIR / "literature_db.xlsx"