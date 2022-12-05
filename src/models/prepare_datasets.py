# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from typing import List


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_file_paths", type=click.Path(), nargs=2)
def prepare_datasets(input_filepath, output_file_paths: List[str]):
    df = pd.read_csv(input_filepath)

    train = df.sample(frac=0.75, random_state=200)
    valid = df.drop(train.index)

    train.to_csv(output_file_paths[0], index=False)
    valid.to_csv(output_file_paths[1], index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    prepare_datasets()
