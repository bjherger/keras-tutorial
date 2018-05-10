#!/usr/bin/env python
"""
coding=utf-8

An example file, highlighting the Keras API, with the titanic data set

"""
import cPickle
import logging
import os

import numpy
import pandas
from keras.callbacks import TensorBoard
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn_pandas import DataFrameMapper

import lib
import models


def main():
    """
    Code entry point

    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # Extract our data
    observations = extract()

    # Transform the data to be input to the model
    x_train, x_test, y_train, y_test, mapper = transform(observations)

    # Model our data
    large_model = model(x_train, x_test, y_train, y_test)

    # Load the relevant assets, for down stream use
    load(mapper, large_model)
    pass


def extract():
    """

     - Extract data from CSV

    :return:
    """
    logging.info('Begin extract')

    # Read files from CSV
    observations = pandas.read_csv('../data/input/titanic.csv')

    # Archive & return
    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return observations


def transform(observations):
    """

     - Convert Sex to boolean male indicator
     - Create train / test split
     - Create SKLearn-Pandas mapper
     - Train SKLearn
     - Transform train and test data

    :param observations:
    :type observations: pandas.DataFrame
    :return:
    """
    logging.info('Begin transform')

    # Convert Sex field into boolean male indicator
    observations['male'] = observations['Sex'] == 'male'
    logging.info('Converted Sex to binary class. Value counts: {}'.format(observations['male'].value_counts()))

    # Split into train / test split
    mask = numpy.random.rand(len(observations)) < 0.8
    observations_train = observations[mask]
    observations_test = observations[~mask]

    logging.info('Creating dataframe mapper')
    mapper = DataFrameMapper([
        (['Age'], [Imputer(), StandardScaler()]),
        (['SibSp'], [Imputer(), StandardScaler()]),
        (['Parch'], [Imputer(), StandardScaler()]),
        (['male'], [Imputer(strategy='most_frequent')])
    ])

    logging.info('Fitting and transforming training data set')
    x_train = mapper.fit_transform(observations_train)
    y_train = observations_train['Survived'].values

    logging.info('Transforming response data set')
    x_test = mapper.transform(observations_test)
    y_test = observations_test['Survived'].values

    # Archive & return
    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return x_train, x_test, y_train, y_test, mapper


def model(x_train, x_test, y_train, y_test):
    """

     - Train multiple models, return a trained model

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    logging.info('Begin model')

    # Baseline model
    baseline_model = models.baseline()
    baseline_model.fit(x_train, y_train, epochs=20, validation_split=.3,
                       callbacks=[TensorBoard(log_dir=os.path.expanduser('~/.logs/baseline'))])

    # Small model
    intermediate_model = models.small()
    intermediate_model.fit(x_train, y_train, epochs=20, validation_split=.3,
                           callbacks=[TensorBoard(log_dir=os.path.expanduser('~/.logs/small'))])

    # Large
    large_model = models.large()
    large_model.fit(x_train, y_train, epochs=20, validation_split=.3,
                    callbacks=[TensorBoard(log_dir=os.path.expanduser('~/.logs/large'))])

    # Archive & return
    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')

    return large_model


def load(mapper, large_model):
    """

     - Save mapper to pkl file
    - Save large model to h5py file

    :param mapper: Mapper, to translate pandas dataframe to usable numpy matrix
    :type mapper: DataFrameMapper
    :param large_model: A trained keras model
    :type large_model: keras.Model

    :return:
    """
    logging.info('Begin load')

    # Save mapper to file
    cPickle.dump(mapper, open('../data/output/mapper.pkl', 'w+'))

    # Save model to file
    large_model.save('../data/output/large_model.h5py', 'w+')

    # Archive & return
    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
