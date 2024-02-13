# Transforms

The base class for all transformations is Transform. This class can be found in the package torchchornos.transforms.base_transforms.

To create a new transform this class has has to be inherited from. The new Transform has to implement the methods: `_fit`, `_tranform`, `_inverse`, `_repr`.

The parameters of all new transforms have to be a tensor containint the data, and a tensor containing the labels. If the dataset has no labels `None` is passed as labels. The transform has to return `None` as well. The logic for splitting the return values of the transform is in `Transform.transform()`.

# Datasets

## BaseDataset

This is a simple implementation of the torch Datset. The main difference to the the class is having a `data` and a `targets` attribure. Mainly this differs in returning always a value for target. When no target array is present, in other words it is `None` the dataset returns `None` as value for targets. This behaviour is used in the different transforms and to ensure a equal behaviour.

## PrepareableDataset

It is possible to remove the targets of a dataset, so they do not get returned when calling `__getItem()__`. This is done with with parameter, `return_labels` or by adding the transform `RemoveLabels()` to the list of transforms. When a label is `None` when it would be returned, it instead does not get returned and only the data is retuned.
