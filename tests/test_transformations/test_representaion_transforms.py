import torch
import numpy as np

from torchchronos.transforms.representation_transformations import LabelTransform, ComplexToPolar, PolarToComplex, CombineToComplex, SplitComplexToRealImag
from torchchronos.transforms.format_conversion_transforms import ToTorchTensor

def test_label_transform():
    transform = LabelTransform()

    data = torch.randn((10, 1, 10), dtype=torch.float32).numpy()
    labels = torch.tensor([1, 5, 2, 1, 4, 1, 9, 0, 5, 1])

    transform.fit(data, labels)
    data_transformed, labels_transformed = transform(data, labels)


    assert np.allclose(data, data_transformed)
    assert np.allclose(labels_transformed, np.array([1, 4, 2, 1, 3, 1, 5, 0, 4, 1]))

    inverse_transform = ~transform
    assert isinstance(inverse_transform, LabelTransform)
    assert inverse_transform.label_map == {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 9}
    

def test_complex_to_polar():
    transform = ComplexToPolar()
    complex_numbers = torch.tensor([3 + 4j, 2 - 2j, -1 + 2j])

    expected_r = torch.tensor([5.0, 2.8284, 2.2361])
    expected_theta = torch.tensor([0.9273, -0.7854, 2.0344])

    polar_coords = transform.transform(complex_numbers)

    assert torch.allclose(polar_coords[:, 0], expected_r, atol=1e-4)
    assert torch.allclose(polar_coords[:, 1], expected_theta, atol=1e-4)

def test_polar_to_complex():
    transform = PolarToComplex()
    polar_coords = torch.tensor([[5.0, 0.9273], [2.8284, -0.7854], [2.2361, 2.0344]]).reshape(3, 2, 1)

    complex_numbers = transform.transform(polar_coords)

    expected_complex_numbers = torch.tensor([3 + 4j, 2 - 2j, -1 + 2j]).reshape(3, 1, 1)

    assert torch.allclose(complex_numbers, expected_complex_numbers, atol=1e-4)

def test_combine_to_complex():
        transform = CombineToComplex()
        data = torch.tensor([[1,2,3,4],[5,4,3,2],[1,6,8,9]]).reshape(3, 1, 4)

        expected_complex = torch.tensor([[[1 + 2j, 3 + 4j]], [[5 + 4j, 3 + 2j]], [[1 + 6j, 8 + 9j]]])
        complex_numbers = transform.transform(data)

        assert torch.allclose(complex_numbers, expected_complex)

def test_split_complex_to_real_imag():
        transform = SplitComplexToRealImag()
        complex_data = torch.tensor([[[1 + 2j, 3 + 4j]], [[5 + 4j, 3 + 2j]], [[1 + 6j, 8 + 9j]]])

        expected = torch.tensor([[1,2,3,4],[5,4,3,2],[1,6,8,9]], dtype=torch.float32).reshape(3, 1, 4)

        data_split = transform.transform(complex_data)
        assert torch.allclose(data_split, expected)
        