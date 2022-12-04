# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def merge_csv(data: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    # Team 1
    df = data.merge(feat, left_on=['map_id', 'team1_id'],
                    right_on=['map_id', 'team_id'],
                    suffixes=('', '_y'), how='inner').drop(['map_name_y', 'team_id'], axis=1)
    # Team 2
    df = df.merge(feat, left_on=['map_id', 'team2_id'],
                  right_on=['map_id', 'team_id'],
                  how='inner', suffixes=('_t1', '_t2')).drop(['map_name_t2', 'team_id'], axis=1)
    # Rename
    df = df.rename(columns={'map_name_t1': 'map_name'}).reset_index(drop=True)
    return df


def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    drop_id = [col for col in df.columns if 'id' in col.lower() and 'team' not in col.lower()]
    df = df.drop(columns=drop_id, axis=1).fillna(0)
    return df


@click.command()
@click.argument('file_df', type=click.Path(exists=True))
@click.argument('file_feat', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def make_dataset(file_df: str, file_feat: str, output_filepath: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load Data
    df = pd.read_csv(file_df)
    feat = pd.read_csv(file_feat)
    # Clear Data
    df = merge_csv(df, feat)
    df = clear_data(df)
    df.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_dataset()
