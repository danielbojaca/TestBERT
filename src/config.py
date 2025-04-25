from pathlib import Path

# Path to the current file
src_dir = Path(__file__).parent / Path("..")
src_dir = src_dir.resolve()

PATHS = {
    "training_data_folder": Path(src_dir, "../src/data/training_data/").resolve(),
    "tokenizer_folder": Path(src_dir, "../src/data/tokenizers/").resolve(),
    "analisis_folder": Path(src_dir, "../src/data/analisis/").resolve(),
}
