import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
import os


def split_and_save_data(csv_path:str, train_output_path: str, test_output_path: str) -> None:
    """Splits the dataset and saves train/test splits."""
    print("Splitting into train and test...")
    df = pd.read_csv(csv_path)
    df["new_tags"] = df["new_tags"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    # check if path exists, if not create it
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

def preprocess_text_fields(row: pd.Series) -> str:
    return (
        str(row.get("prob_desc_description", "")) + " " +
        str(row.get("prob_desc_notes", "")) + " " +
        str(row.get("source_code", ""))
    )