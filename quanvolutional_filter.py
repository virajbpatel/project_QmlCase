# To be run in Python 3.7, all other files work with Python 3.9

# Import releveant packages
import torch
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.layers import U3CU3Layer0
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os

class CustomImageDataset(Dataset):
    '''
    A class to represent a PyTorch Dataset object based on a custom dataset.

    ...

    Attributes
    ----------
    img_labels : pandas.Dataframe
        labels for each image in the dataset
    img_dir : str
        directory where images are stored
    transform : torch.transforms
        set of transformations to apply to each image
    target_transform : torch.transforms
        set of transformations to apply to each label
    
    Methods
    -------
    __len__(self):
        Returns length of dataset
    __getitem__(self, idx):
        Returns the image and label at index idx
    '''

    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None) -> None:
        '''
        Returns a PyTorch Dataset object
            Parameters:
                annotations_file (str): file name for csv file with labels for the images
                img_dir (str): directory where images are stored
                transform (torchvision.transforms, optional): transformations to apply to images (default is None)
                target_transform (torchvision.transforms, optional): transformations to apply to labels (defualt is None)
            Returns:
                image (torch.tensor): tensor object representing each image
        '''
        self.img_labels = pd.read_csv(annotations_file, index_col = [0])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        Returns length of dataset
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        Returns image and label at index idx
            Parameters:
                idx (int): index of image
            Returns:
                image (torch.Tensor): image at index idx
                label (torch.Tensor): label at index idx
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class QuanvolutionalFilter(tq.QuantumModule):

    '''
    A class that creates a quanvolutional filter using a random quantum circuit. This is not trainable.

    ...

    Attributes
    ----------
    n_wires : int
        number of qubits in the quanvolutional circuit
    img_size : int
        width of square image in pixels
    processor : qiskit.IBMQ.provider
        IBM Q provider if using a real quantum device
    q_device : torchquantum.QuantumDevice
        quantum circuit
    encoder : torchquantum.GeneralEncoder
        protocol to encode classical data into the quantum circuit
    q_layer : torchquantum.RandomLayer
        unitary operation to perform convolution
    measure : torchquantum.MeasureAll
        measurement function in Pauli-Z basis

    Methods
    -------
    forward(self, x, use_qiskit):
        runs the quanvolutional algorithm over input image
    '''

    # Initialise a 4-qubit quantum circuit to encode pixels in, and then perform quanvolution with layers of 
    # random gates
    def __init__(self, n_qubits, img_size, pool = True, processor = None) -> None:
        '''
        Creates a quanvolutional filter with a parameterised quantum circuit
            Parameters:
                n_qubits (int): number of qubits in the quantum circuit, also the number of pixels in the quanvolutional filter
                img_size (int): height or width of square input image in pixels
                pool (bool, optional): True if pooling is included in quanvolutional layer (default is True)
                processor (qiskit.IBMQ.provider, optional): IBM-Q provider if using a real device (default is None)
        '''
        super().__init__()
        self.pool = pool
        self.processor = processor
        self.n_wires = n_qubits
        self.img_size = img_size
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        encoding_list = []
        for i in range(self.n_wires):
            encoding = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            encoding_list.append(encoding)
        self.encoder = tq.GeneralEncoder(encoding_list)
        self.arch = None
        self.q_layer = tq.RandomLayer(n_ops = 8, wires = list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.kernel_size = int(np.sqrt(self.n_wires))

    def forward(self, x, use_qiskit = False):
        '''
        Passes the quanvolutional filter over the input image
            Parameters:
                x (torch.Tensor): input image
                use_qiskit (bool, optional): True if using a real quantum device (default is False)
            Returns:
                result (torch.Tensor): quanvolved image
        '''
        bsz = 1
        size = self.img_size
        stride = self.kernel_size
        if self.pool:
            step = stride
            stop = size-1
        else:
            step = 1
            stop = size - stride + 1
        x = x.view(bsz, size, size)

        data_list = []

        for c in range(0, stop, step):
            row = []
            for r in range(0, stop, step):
                pixels = []
                for i in range(stride):
                    for j in range(stride):
                        pixels.append(x[:,c+i,r+j])
                data = torch.transpose(torch.cat(tuple(pixels)).view(self.n_wires,bsz), 0, 1)
                if use_qiskit:
                    self.set_qiskit_processor(self.processor)
                    data = self.qiskit_processor.process_parameterized(
                        self.q_device, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(self.q_device, data)
                    self.q_layer(self.q_device)
                    data = self.measure(self.q_device)

                row.append(data.view(bsz, self.n_wires))
            data_list.append(torch.stack(row))
        data_list = torch.stack(data_list)
        data_list = torch.transpose(torch.squeeze(data_list), 0, 2).float()
        result = data_list
        return result

class TrainableQuanvolutionalFilter(QuanvolutionalFilter):

    '''
    A class that creates a quanvolutional filter using a parameterised quantum circuit. This is trainable.

    ...

    Attributes
    ----------
    pool : bool
        decides whether the pooling is done within the quanvolutional layer or not
    n_wires : int
        number of qubits in the quanvolutional circuit
    img_size : int
        width of square image in pixels
    processor : qiskit.IBMQ.provider
        IBM Q provider if using a real quantum device
    q_device : torchquantum.QuantumDevice
        quantum circuit
    encoder : torchquantum.GeneralEncoder
        protocol to encode classical data into the quantum circuit
    arch : dict
        hyper-parameters for parameterised quantum circuit
    q_layer : torchquantum.RandomLayer
        unitary operation to perform convolution
    measure : torchquantum.MeasureAll
        measurement function in Pauli-Z basis
    kernel_size : int
        width or height of square kernel in pixels, equal to square root of n_wires

    Methods
    -------
    forward(self, x, use_qiskit):
        runs the quanvolutional algorithm over input image
    '''

    def __init__(self, n_qubits, img_size, pool = True, processor = None) -> None:
        '''
        Creates a quanvolutional filter with a parameterised quantum circuit
            Parameters:
                n_qubits (int): number of qubits in the quantum circuit, also the number of pixels in the quanvolutional filter
                img_size (int): height or width of square input image in pixels
                pool (bool, optional): True if pooling is included in quanvolutional layer (default is True)
                processor (qiskit.IBMQ.provider, optional): IBM-Q provider if using a real device (default is None)
        '''
        super().__init__(n_qubits, img_size, pool, processor)
        self.arch = {'n_wires': self.n_wires, 'n_blocks': 3, 'n_layers_per_block': 2}
        self.q_layer = U3CU3Layer0(self.arch)


class QuantumClassifier(torch.nn.Module):

    '''
    Creates a quantum classifier that directly takes in the image

    ...

    Attributes
    ----------
    n_wires : int
        number of qubits in circuit
    processor : qiskit.IBMQ.processor
        IBM-Q processor of real quantum device
    q_device : torchquantum.QuantumDevice
        quantum circuit
    encoder : torchquantum.GeneralEncoder
        function that encodes classical data into quantum circuit
    arch : dict
        hyper-parameters of quantum circuit
    ansatz : torchquantum.layers
        parameterised quantum circuit used for classification
    measure : torchquantum.MeasureAll
        function that measures circuit in Pauli-Z basis

    Methods
    -------
    forward(self, x, use_qiskit):
        parses image into quantum classifier
    '''

    def __init__(self, n_qubits, encoding, processor = None) -> None:
        '''
        Creates quantum classifier
            Parameters:
                n_qubits (int): number of qubits in quantum circuit
                encoding (str): encoding protocol from torchquantum
                processor (qiskit.IBMQ.processor, optional): processor of real quantum device (default is None)
        '''
        super().__init__()
        self.n_wires = n_qubits
        self.processor = processor
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[encoding])
        self.arch = {'n_wires': self.n_wires, 'n_blocks': 8, 'n_layers_per_block': 2}
        self.ansatz = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit = False):
        '''
        Parses image into quantum classifier
            Parameters:
                x (torch.Tensor): input image
                use_qiskit (bool, optional): True is using a real quantum device (default is False)
            Returns:
                x (torch.Tensor): predicted label from classifier
        '''
        if use_qiskit:
            self.set_qiskit_processor(self.processor)
            x = self.qiskit_processor.process_parameterised(
                self.q_device, self.encoder, self.q_layer, self.measure, x
            )
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)
        return x

class QFC(tq.QuantumModule):

    '''
    Creates a quantum fully connected layer that takes a feature vector.

    ...

    Attributes
    ----------
    n_wires : int
        number of qubits in circuit
    processor : qiskit.IBMQ.processor
        IBM-Q processor of real quantum device
    q_device : torchquantum.QuantumDevice
        quantum circuit
    encoder : torchquantum.GeneralEncoder
        function that encodes classical data into quantum circuit
    arch : dict
        hyper-parameters of quantum circuit
    q_layer : torchquantum.layers
        parameterised quantum circuit used for classification
    measure : torchquantum.MeasureAll
        function that measures circuit in Pauli-Z basis

    Methods
    -------
    forward(self, x, use_qiskit):
        parses feature vector into quantum fully connected layer
    '''

    def __init__(self, n_qubits, encoding, processor = None) -> None:
        '''
        Creates quantum fully connected layer
            Parameters:
                n_qubits (int): number of qubits in quantum circuit
                encoding (str): encoding protocol from torchquantum
                processor (qiskit.IBMQ.processor, optional): processor of real quantum device (default is None)
        '''
        super().__init__()
        self.n_wires = n_qubits
        self.processor = processor
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[encoding])
        self.arch = {'n_wires': self.n_wires, 'n_blocks': 4, 'n_layers_per_block': 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit = False):
        '''
        Parses feature vector into quantum fully connected layer
            Parameters:
                x (torch.Tensor): input feature vector
                use_qiskit (bool, optional): True is using a real quantum device (default is False)
            Returns:
                data (torch.Tensor): predicted label from classifier
        '''
        data = x
        if use_qiskit:
            self.set_qiskit_processor(self.processor)
            data = self.qiskit_processor.process_parameterised(
                self.q_device, self.encoder, self.q_layer, self.measure, data
            )
        else:
            self.encoder(self.q_device, data)
            self.q_layer(self.q_device)
            data = self.measure(self.q_device)
        return data