from pathlib import Path

# Path to the current file
src_dir = Path(__file__).parent / Path("..")
src_dir = src_dir.resolve()

PATHS = {
    "training_data_folder": Path(src_dir, "../datasets/").resolve(),
    "tokenizer_folder": Path(src_dir, "data/tokenizers/").resolve(),
    "analisis_folder": Path(src_dir, "data/analisis/").resolve(),
    "models_folder": Path(src_dir, "data/modelos/").resolve(),
    "trainer_folder": Path(src_dir, "data/trainer/").resolve(),
}
