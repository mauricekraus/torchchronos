Welcome to torchchronos!
========================

Introduction
============

Why torchchronos
----------------
TorchChronos is a library to make handeling time series data easy with PyTorch and Lightning. There are alreads some libraries for handeling
time series data such as SkTime or Darts, however those libraries do not work together with PyTorch.

The goal of this library is to bridge that gap, by implementing a library that takes time series data and makes processing and using it in
a PyTorch environment easy. TorchChronos comes with several features, such as loading many datasets and being able to get a lightning module
for an easy use.

The library is divided into 3 main parts:

.. toctree::
   :maxdepth: 1

   datasets
   transforms
   lightning

+ Datasets: This module allows you to access and work with various time series datasets.
+ Transforms: This module provides a range of transformations that can be applied to the data.
+ Lightning: This module supports the creation and use of Lightning modules, enhancing the integration with PyTorch Lightning.

Installation
============
The library can be installed via pip. Just run

.. code-block:: bash
   
   pip install torchchronos


Example Code
============
.. code-block:: python
   :linenos:

   from torchchronos.transforms import Scale
   from torchchronos.datasets import MonashForcastingDataset
   from torchchronos.lightning import DatasetDataModule
   # Load the dataset
   
   dataset = MonashForcastingDataset(
       name="sunspot_dataset_without_missing_values", transform=scale_transform
   )

   dataset.prepare()
   dataset.load()
   
   # Apply a transformation on it
   scale_transform = Scale(10)
   scale_transform.fit(dataset)

   # Create a DataModule
   ddm = DatasetDataModule(dataset, val=0.2, test=0.2)
   ddm.setup("fit")
   ddm.setup("test")

   train_loader = ddm.train_dataloader()
   test_loader = ddm.test_dataloader()





