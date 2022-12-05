# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import joblib as jb
from pathlib import Path
from typing import List
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def train(input_paths: List[str], output_path: str):
    train_df = pd.read_csv(input_paths[0])
    valid_df = pd.read_csv(input_paths[1])

    x_train = train_df.drop('who_win', axis=1)
    y_train = train_df['who_win']
    x_holdout = valid_df.drop('who_win', axis=1)
    y_holdout = valid_df['who_win']

    searcher = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear'), [{"C": np.logspace(-5, 1, 15)}],
                            scoring="roc_auc", cv=10, n_jobs=-1)
    searcher.fit(x_train, y_train)
    logreg = LogisticRegression(penalty='l2', solver='liblinear', C=searcher.best_params_["C"])
    jb.dump(logreg, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    train()
