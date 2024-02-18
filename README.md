<div style="text-align:center;">
<img src="docs/penguin_in_huddles_dall_e.png" alt="penguin in huddle" width="450px"/>
</div>

# huddles

<p align="center">
<!-- <a href="https://pypi.org/project/black/"><img alt="PyPI" src="https://img.shields.io/pypi/v/black"></a> -->
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/ydata-profiling)](https://pypi.org/project/ydata-profiling/)  -->
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) -->
</p>

> *Data, like penguins, is happier when huddled together*

 *huddles* helps you explore patterns in your data.

 The core functionality is the `embedding_exploration_plot()` function, which will allow you to navigate a 2D representation of your data and bring to light the underlying patterns.

Note that *huddles* is only a visualisation library. The dimensionality reduction or embedding calculations must be done with adequate third party tools. For example:

- [PCA decomposition](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) (scikit-learn)
- [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE) (scikit-learn)
- [UMAP](https://github.com/lmcinnes/umap)

## Installation

```bash
pip install huddles
```

The library has been tested with Python 3.11.

## Usage

This is an example of how to use the *huddles* library to explore the Iris dataset.

```python  #
from sklearn import datasets
from sklearn.manifold import TSNE
import pandas as pd
import huddles

# Load the Iris dataset
iris = datasets.load_iris()
input_data = pd.DataFrame(iris.data, columns=iris.feature_names)
targets = pd.Series([iris.target_names[x] for x in iris.target]).rename("species")

# Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2)

# Preparing the DataFrame
embeddings = tsne.fit_transform(input_data)
full_dataset = pd.concat([input_data, targets], axis=1)

# Explore the embedding with huddles
huddles.embedding_exploration_plot(embeddings, full_dataset, hue='species')')
```

**TODO: ADD GIF**

## Backlog of features

- [ ] Add an optional 2d kde plot to the scatter plot to visualise the which the value of the target in the different clusters
- [ ] Add an extra plot that compares feature by feature the distribution of the selection vs. the whole population

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

Inspired by [cluestar](https://github.com/koaning/cluestar)
