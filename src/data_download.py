from pathlib import Path
import time
import kagglehub
import shutil

destination_raw_data = Path("data/raw/paysim.csv")

if not destination_raw_data.exists():
    print("Downloading dataset...")
    
    t0 = time.perf_counter()

    cache_path = kagglehub.dataset_download("ealaxi/paysim1")

    t1 = time.perf_counter()

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Assume the filename encoutered when starting the project
    source = Path(cache_path) / "PS_20174392719_1491204439457_log.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Expected source file not found: {source}. "
            "The Kaggle dataset structure may have changed."
        )

    # change the filename to something more readable
    shutil.copy2(source, destination_raw_data)

    t2 = time.perf_counter()

    print(f"Copied data to {destination_raw_data}")
    print(f"Download: {t1-t0:.2f}s")
    print(f"Copy CSV: {t2-t1:.2f}s")
else:
    print("Dataset already exists. Skipping download.")