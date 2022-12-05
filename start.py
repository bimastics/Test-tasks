import src

TRAIN_DATA_PATH = "data/raw/train.csv"
TEST_DATA_PATH = "data/raw/test.csv"
FEATURES_DATA_PATH = "data/raw/players_feats.csv"

MAKE_DATASET_PATH = "data/interim/make_dataset.csv"
FEATURED_DATA_PATH = "data/processed/data_featured.csv"

if __name__ == "__main__":
    print(1)
    # src.make_dataset(TRAIN_DATA_PATH, FEATURES_DATA_PATH, MAKE_DATASET_PATH)

    src.build_features(MAKE_DATASET_PATH, FEATURED_DATA_PATH, "train")
    # src.build_features("data/interim/test.csv", "data/processed/test.csv", "transform")
    # src.predict_model(["data/processed/test.csv", "models/model.clf"], "data/predicts/test_out.csv")
