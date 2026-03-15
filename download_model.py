"""
download_model.py
=================
Downloads model files from Google Drive if not present locally.
Called at the top of app.py before loading any models.
"""

import os
import gdown

FILES = {
    "model_regression.keras":  "1LC8DXiBRSAi3ELe_KP77wd_gY3HXxGRi",
    "model_classifier.keras":  "1x1WYryKiw5rcWWbpMVnafGuNGvGdQOJ1",
    "scaler_ret.pkl":          "1drwEskv90EmMQ_FtJo0SOenG13v7vehy",
    "scaler_features.pkl":     "1c2KzbgMCbHkOt3VCsMPcXWVfMvoGDu4q",
    "feature_cols.pkl":        "1erTu-0IcZzhBmv67pqAzofYKTbtvJ8o8",
    "cls_threshold.pkl":       "19GwiqbXOULftOFbYY4qRKZ0-0rWffAw3",
}

def download_if_missing():
    for filename, file_id in FILES.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            gdown.download(id=file_id, output=filename, quiet=False)
            print(f"  ✓ {filename}")

if __name__ == "__main__":
    download_if_missing()