# Transforms

The base class for all transformations is Transform. This class can be found in the package torchchornos.transforms.base_transforms.

To create a new transform this class has has to be inherited from. The new Transform has to implement the methods: `_fit`, `_tranform`, `_inverse`, `_repr`.

Note that the ``_transform()`` method, has to return one of the following tuples:
```python
_transform(Tensor, None) -> tuple[Tensor, None]
_transform(Tensor, Tensor) -> tuple[Tensor, Tensor]
```

It is possible to work without labels, just never pass a value to a ``targets`` attribute, `.transform()` will return only one value
# Datasets

## BaseDataset

This is a simple implementation of the torch Datset. The main difference to the the class is having a `data` and a `targets` attribure. Mainly this differs in returning always a value for target. This dataset is mainly used for transforming whole datasets and returning them in a ``BaseDatast``

## PrepareableDataset

This class extends the ``torch.Dataset`` class with a prepare and load method. The ``prepare()`` method is for actions that have to be done before loading the dataset, e.g. download data or check if paths are correct. The ``load()`` method is for loading the data into memory, this has to be done before the dataset is usable. Note that the interiting classes have to fit the transforms, this is not done in this class.

## AeonDatasets
The classes are for loading and using datasets from the aeon framework. In the prepare-step the data is downloaded, and in the load-step the data is loaded, and the transformations fit onto the data.

## CachedDataset
This class is for loading data that is stored somewhere. The prepare-step checks whether the path to the files exists. The load-step loads the data. Currently only numpy data is supported.