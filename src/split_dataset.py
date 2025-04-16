from utils import split_and_save_data
from config import DATA_PATH, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH

if __name__ == "__main__":
    
    # Split and save data
    split_and_save_data(DATA_PATH, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH)