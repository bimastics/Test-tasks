# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
import pandas as pd
from typing import List
from pathlib import Path

USE_COLUMNS = ['scaling__index', 'who_win']


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def predict_model(input_paths: List[str], output_path: str):
    test_df = pd.read_csv(input_paths[0])
    with open(Path(os.getcwd(), input_paths[1]), 'rb') as f:
        model = pickle.load(f)

    # x_holdout = test_df.drop('who_win', axis=1)
    # y_holdout = test_df['who_win']

    # y_predicted = model.predict(x_holdout, num_iteration=model.best_iteration)
    # score = pd.DataFrame(
    #     dict(
    #         mae=mean_absolute_error(y_holdout, y_predicted),
    #         rmse=mean_squared_error(y_holdout, y_predicted),
    #         roc=roc_auc_score(y_holdout, y_predicted)
    #     ),
    #     index=[0],
    # )
    # score.to_csv(output_path[1], index=False)
    y_predicted = model.predict(test_df)
    test_df['who_win'] = y_predicted
    test_df[USE_COLUMNS].to_csv(output_path, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    predict_model()
