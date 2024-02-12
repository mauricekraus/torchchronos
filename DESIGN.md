The base class for all transformations is Transform. This class can be found in the package torchchornos.transforms.base_transforms.

To create a new transform this class has has to be inherited from. The new Transform has to implement the methods: `_fit`, `_tranform`, `_inverse`, `_repr`.

The parameters of all new transforms have to be a tensor containint the data, and a tensor containing the labels. If the dataset has no labels `None` is passed as labels. The transform has to return `None` as well. The logic for splitting the return values of the transform is in `Transform.transform()`.
