# Transforms the first 200 lines of processed_data/15Second.parquet to a csv file for you to view the format and build an algorithm that accepts that format.
import pandas as pd
from pathlib import Path

def main():
    src = Path("algorithm1/artefacts/artefacts/feature_matrix.parquet")
    dst = Path("sample_feature_matrix.csv")
    rows = 200

    if not src.exists():
        print(f"❌  Parquet file not found: {src}")
        return

    print(f"📥 Reading {src} …")
    # Use pyarrow engine; fast & memory-efficient
    df = pd.read_parquet(src, engine="pyarrow")[:rows]

    print(f"💾 Writing first {rows} rows to {dst} …")
    df.to_csv(dst, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()