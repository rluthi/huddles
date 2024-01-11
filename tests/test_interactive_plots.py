import numpy as np
import pandas as pd
import pytest

import altair

from huddles import embedding_exploration_plot


@pytest.fixture
def random_embeddings_and_data():
    embeddings = np.random.rand(10, 2)
    dataset = pd.DataFrame({"num": np.random.rand(10), "str_cats": ["a", "b", "c", "a", "a", "b", "c", "b", "a", "d"]})
    return embeddings, dataset


# Testing a walkthrough with valid data (visual tests must always also be done manually)
def test_valid_input_with_defaults(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    result = embedding_exploration_plot(embeddings, dataset)
    assert isinstance(result, type(altair.HConcatChart()))


def test_valid_input_with_numerical_hue(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    result = embedding_exploration_plot(embeddings, dataset, hue="num")
    assert isinstance(result, type(altair.HConcatChart()))


def test_valid_input_with_categorical_hue(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    result = embedding_exploration_plot(embeddings, dataset, hue="str_cats")
    assert isinstance(result, type(altair.HConcatChart()))


# Testing the errors cases (Are they correctly raised?)
def test_embeddings_invalid_column_count(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    embeddings = np.random.rand(10, 3)  # 3 columns instead of 2
    with pytest.raises(ValueError):
        embedding_exploration_plot(embeddings, dataset)


def test_mismatch_rows(random_embeddings_and_data):
    embeddings, _ = random_embeddings_and_data
    dataset = pd.DataFrame({"a": np.random.rand(9), "b": np.random.rand(9)})  # 9 rows instead of 10
    with pytest.raises(ValueError):
        embedding_exploration_plot(embeddings, dataset)


def test_invalid_hue(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    with pytest.raises(ValueError):
        embedding_exploration_plot(embeddings, dataset, hue="c")  # 'c' is not in the dataset
