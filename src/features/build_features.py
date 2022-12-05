# -*- coding: utf-8 -*-
import os
import click
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV


class CreateFeatures:
    def __init__(self, df: pd.DataFrame):
        X = df.drop(columns=(['who_win']), axis=1)
        self.col_tf = self.fit_col_tf_data(X)
        X = self.col_tf.transform(X)
        self.kmeans_10 = KMeans(n_clusters=10, random_state=33)
        self.kmeans_10.fit(X)
        self.kmeans_3 = KMeans(n_clusters=3, random_state=33)
        self.kmeans_3.fit(X)
        self.is_train = True
        self.use_features = None

    @staticmethod
    def fit_col_tf_data(df: pd.DataFrame) -> ColumnTransformer:
        categorical_features = df.select_dtypes(include='object').columns
        numeric_features = df.select_dtypes(exclude='object').columns

        return ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ('scaling', StandardScaler(), numeric_features),
        ]).fit(df)

    @staticmethod
    def reduce_dims_to_nd_space_with_pca(df: pd.DataFrame, n: int) -> pd.DataFrame:
        pca = PCA(n_components=n)
        components = pca.fit_transform(df)
        return pd.DataFrame(data=components, columns=['component_' + str(i) for i in range(1, n + 1)])

    @staticmethod
    def reduce_dims_to_nd_space_with_tsne(df: pd.DataFrame, n: int) -> pd.DataFrame:
        tsne = TSNE(n_components=n, learning_rate=250, random_state=42)
        components = tsne.fit_transform(df)
        return pd.DataFrame(data=components, columns=['component_' + str(i) for i in range(1, n + 1)])

    def drop_features(self, df: pd.DataFrame, y: pd.Series) -> np.array:
        searcher = GridSearchCV(Lasso(), [{"alpha": np.logspace(-4, -2, 25)}],
                                scoring="roc_auc", cv=10, n_jobs=-1)
        searcher.fit(df, y)
        model = Lasso(alpha=searcher.best_params_["alpha"]).fit(df, y)
        self.is_train = False
        self.use_features = df.columns[model.coef_ != 0].to_list() + ['who_win']

    def create_new_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        y = df[['who_win']]
        X = df.drop(columns=(['who_win']), axis=1)
        # X = self.col_tf.transform(X)
        X = pd.DataFrame(self.col_tf.transform(X), columns=self.col_tf.get_feature_names_out())

        # Понижение размерности
        # components_3d_pca = self.reduce_dims_to_nd_space_with_pca(X, n=3)
        components_3d_tsne = self.reduce_dims_to_nd_space_with_tsne(X, n=3)

        # Создаём новые признаки
        labels_clast_10 = pd.Series(self.kmeans_10.predict(X), name='clusters_2')
        labels_clast_3 = pd.Series(self.kmeans_3.fit_predict(X), name='clusters_3')

        # Склейка данных
        clusters_3_dummies = pd.get_dummies(labels_clast_3, drop_first=True, prefix='clusters_3')
        clusters_10_dummies = pd.get_dummies(labels_clast_10, drop_first=True, prefix='clusters_10')

        df_ext = pd.concat([X, components_3d_tsne, clusters_3_dummies,
                            clusters_10_dummies, y], axis=1)
        if self.is_train:
            self.drop_features(df_ext.drop(columns=(['who_win']), axis=1), df_ext['who_win'])
        return df_ext[self.use_features]


@click.command()
@click.argument('file_df', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('type', type=click.STRING)
def build_features(file_df: str, output_filepath: str, type: str = 'train'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(file_df)
    if type == 'train':
        generator = CreateFeatures(df)
        with open(Path(os.getcwd(), 'models/features.pickle'), 'wb') as f:
            pickle.dump(generator, f)
    elif type == 'transform':
        with open(Path(os.getcwd(), 'models/features.pickle'), 'rb') as f:
            generator = pickle.load(f)

    generator.create_new_feat(df).to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    build_features()
