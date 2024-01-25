from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:
        x_raw = np.frombuffer(f.read(), 'B', offset=16) # the magic number, num_image, num_row, num_col are the offset (4 * 4 bytes)
    x_reshaped = x_raw.reshape(-1, 784).astype('float32')
    x_reshaped = x_reshaped/255

    with gzip.open(label_filename, 'rb') as f:
        y_raw = np.frombuffer(f.read(), 'B', offset=8) # the magic number, num_image, num_row, num_col are the offset (4 * 4 bytes)
    y_reshaped = y_raw.reshape(-1).astype('uint8')
    print("tianhao debug", type(y_reshaped))
    return x_reshaped, y_reshaped
    ### END YOUR CODE

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms=transforms)
        self.images, self.labels = parse_mnist(image_filename=image_filename, label_filename=label_filename)
        self.len = self.labels.shape[0]

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index].reshape((28, 28, 1))
        transformed_img = self.apply_transforms(img)
        return (transformed_img, self.labels[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.len
        ### END YOUR SOLUTION