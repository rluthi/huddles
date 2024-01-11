import numpy as np
import pandas as pd
import pytest
import altair

from huddles import create_exploration_plot


@pytest.fixture
def random_embeddings_and_data():
    embeddings = np.random.rand(10, 2)
    dataset = pd.DataFrame({"a": np.random.rand(10), "b": np.random.rand(10)})
    return embeddings, dataset


# TODO: test with numerical hue
# TODO: test with categorical hue
# TODO: test with Null as hue


def test_valid_input(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    result = create_exploration_plot(embeddings, dataset)
    assert isinstance(result, altair.vegalite.v4.api.VConcatChart)


def test_embeddings_invalid_column_count(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    embeddings = np.random.rand(10, 3)  # 3 columns instead of 2
    with pytest.raises(ValueError):
        create_exploration_plot(embeddings, dataset)


def test_mismatch_rows(random_embeddings_and_data):
    embeddings, _ = random_embeddings_and_data
    dataset = pd.DataFrame({"a": np.random.rand(9), "b": np.random.rand(9)})  # 9 rows instead of 10
    with pytest.raises(ValueError):
        create_exploration_plot(embeddings, dataset)


def test_invalid_hue(random_embeddings_and_data):
    embeddings, dataset = random_embeddings_and_data
    with pytest.raises(ValueError):
        create_exploration_plot(embeddings, dataset, hue="c")  # 'c' is not in the dataset
