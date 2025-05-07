#!/bin/bash

FILE="/usr/local/lib/python3.10/dist-packages/diffusers/dynamic_modules_utils.py"

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "File does not exist: $FILE"
    exit 1
fi

# Backup
cp "$FILE" "$FILE.bak"
echo "Backup was created: $FILE.bak"

# Remove 'cached_download' from import lines
sed -i '/from huggingface_hub import/s/cached_download[,]* *//' "$FILE"

# Replace all calls to cached_download() with hf_hub_download()
sed -i 's/cached_download(/hf_hub_download(/g' "$FILE"

echo "Fixing done: cached_download -> hf_hub_download"
