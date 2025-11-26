#!/usr/bin/env bash

# Root directory containing subfolders with parquet files
ROOT_DIR="$1"

if [ -z "$ROOT_DIR" ]; then
    echo "Usage: ./inspect_parquets.sh <root_dir>"
    exit 1
fi

echo "📂 Recursively inspecting parquet files under: $ROOT_DIR"
echo

# Use find to get ALL parquet files in all subdirectories
find "$ROOT_DIR" -type f -name "*.parquet" | while read -r FILE; do

    DIR=$(dirname "$FILE")
    BASENAME=$(basename "$FILE")

    echo "=============================================="
    echo "📁 Directory: $DIR"
    echo "📄 File: $BASENAME"
    echo "=============================================="

    # Use Python to inspect parquet
    python3 - <<EOF
import pandas as pd

file = "$FILE"

try:
    df = pd.read_parquet(file)
    print("🔹 Columns:")
    print(df.columns.tolist())
    print("\n🔸 Head:")
    print(df.head(5))
except Exception as e:
    print(f"❌ Error reading parquet: {e}")
EOF

    echo
done
