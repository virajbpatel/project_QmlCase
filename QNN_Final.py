import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchmetrics

import qiskit
from qiskit import execute, transpile, assemble, BasicAer, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit.visualization import *
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#This part is only needed for my Mac
#import os 
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
For loading images into a proper dataset 
"""
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None) -> None:
        self.img_labels = pd.read_csv(annotations_file, index_col = [0])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

"""
For some tensor manipulation
"""
def to_numbers(tensor_list):
    num_list = []
    for tensor in tensor_list:
        num_list += [tensor.item()]
    return num_list


class QiskitCircuit():
    def __init__(self, n_qubits, backend, shots):
        self.shots = shots
        self.backend = backend
        q=QuantumRegister(n_qubits)
        c=ClassicalRegister(1)
        self.circuit = QuantumCircuit(q,c)
        self._n_qubits = n_qubits
        all_qubits = [i for i in range(n_qubits)]
        self.theta = Parameter('theta')

        self.circuit.rx(np.pi/4, all_qubits)
        self.circuit.ry(self.theta, all_qubits)
        for i in range(n_qubits-1):
            self.circuit.cx(i,i+1)
        self.circuit.ry(self.theta, all_qubits)
        for i in range(n_qubits-1):
            self.circuit.cx(i,i+1)
        self.circuit.measure_all()
    
    def energy_expectation(self, counts, shots, i,j, Cij=-1): #calculate expectation for one qubit pair, for 'measurement' from perceptrons
        expects = 0
        for key in counts.keys():
            perc = counts[key]/shots
            check = Cij*(float(key[i])-1/2)*(float(key[j])-1/2)*perc
            expects += check   
        return [expects] 
    
    def bind(self,parameters): #for assigning neural network parameters to the quantum circuit rotation gate
        self.theta = parameters
        set_done=0
        for i in range(len(self.circuit.data)):
            if self.circuit.data[i][0]._name =='ry':
                self.circuit.data[i][0]._params = [to_numbers(parameters)[set_done]]
            else:
                if self.circuit.data[i-1][0]._name =='ry': 
                    set_done+=1
                else:
                    pass
        return self.circuit
 
    def run(self, i):
        self.bind(i)
        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute(self.circuit,backend,shots=self.shots)
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        return self.energy_expectation(counts, self.shots, 0,1)   


class TorchCircuit(Function):    
    @staticmethod
    def forward(ctx, i):
        NUM_QUBITS = 4
        NUM_SHOTS = 1000
        SIMULATOR = Aer.get_backend('qasm_simulator') 
        if not hasattr(ctx, 'QiskitCirc'):
            ctx.QiskitCirc = QiskitCircuit(NUM_QUBITS, SIMULATOR, shots=NUM_SHOTS)
        
        exp_value = ctx.QiskitCirc.run(i)
        result = torch.tensor([exp_value])
        ctx.save_for_backward(result, i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        SHIFT = 0.9
        forward_tensor, i = ctx.saved_tensors
        input_numbers = i
        gradients = torch.Tensor()
        for k in range(len(input_numbers)):
            shift_right = input_numbers.detach().clone()
            shift_right[k] = shift_right[k] + SHIFT
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - SHIFT

            expectation_right = ctx.QiskitCirc.run(shift_right)
            expectation_left  = ctx.QiskitCirc.run(shift_left)
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])*2
            gradients = torch.cat((gradients, gradient.float()))
        result = torch.Tensor(gradients)
        return (result.float() * grad_output.float()).T

class Net(nn.Module):
    def __init__(self,N_QUBITS,N_CIRCUITS):
        self.nqubits = N_QUBITS
        self.ncircuits = N_CIRCUITS
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(43264,self.nqubits*self.ncircuits*2)
        self.fake_qc = nn.Linear(self.nqubits, self.ncircuits)
        self.qc = TorchCircuit.apply
        self.out = nn.Linear(self.ncircuits, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x) # rescale to [-1, 1] to get sin(theta), cos(theta) for each angle
        x = x.view(-1) # flatten to 1D tensor
        x = torch.cat([(x[i * 2 : i*2 + 2]) / torch.norm(x[i * 2: i*2 + 2]) 
                       for i in range(self.ncircuits*2)], dim=-1) # normalize sin and cos for each angle
        x = torch.stack([torch.atan2(x[i * 2], x[i*2 + 1]) for i in range(self.ncircuits*2)]) # convert to angles
        x = torch.cat([self.qc(x[i * 2 : i*2 + 2]) for i in range(self.ncircuits)], dim=1) # QUANTUM LAYER
        x = self.out(x)

        return F.log_softmax(x,dim=1) 
    
    def predict(self, x):
        # apply softmax
        pred = self.forward(x)
        ans = torch.argmax(pred[0]).item()
        return torch.tensor(ans)

"""
Putting everything together in 1 class to perform training, testing, and saving of data/models
train_loader,val_loader,test_loader not defined in this file, not sure if we should rename these here
"""
class quantum_model():
    def __init__(self, N_QUBITS, N_CIRCUITS, LR, MOM, train_data, val_data, test_data):
        self.NQUBITS = N_QUBITS
        self.NCIRCUITS = N_CIRCUITS
        self.LR = LR
        self.MOM = MOM
        self.trainset = train_data
        self.valset = val_data
        self.testset = test_data

        self.model = Net(self.NQUBITS, self.NCIRCUITS) 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LR,
                        momentum=self.MOM,nesterov=True)
                    
        self.loss = nn.NLLLoss()

        self.loss_list_quantum = []
        self.pq = []
        self.tq = []

    def train(self,EPOCH):
        self.EPOCH = EPOCH
        self.model.train()
        self.accuracy_val=[]
        for epoch in range(self.EPOCH):
            count=0
            total_loss = []
            correct=0
            for batch_idx, (data, target) in enumerate(tqdm(self.trainset, position=0, leave=True)):
                self.optimizer.zero_grad()        
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            
            self.corr_val=0
            self.loss_val=[]
            for batch_idx, (data, target) in enumerate(self.valset):
                val_out = self.model(data)  
                pred = val_out.argmax(dim = 1, keepdim = True)
                self.corr_val += pred.eq(target.view_as(pred)).sum().item()
                loss = self.loss(val_out, target)
                self.loss_val.append(loss.item())
            self.accuracy_quantum_val=self.corr_val/len(self.valset)*100
            self.accuracy_val.append(self.accuracy_quantum_val)
            self.loss_list_quantum.append(sum(self.loss_val)/len(self.loss_val))
            print('Quantum Training [{:.0f}%]\t Val Loss: {:.4f} \t Val Accuracy: {:.1f}'.format(100. * (epoch + 1) / self.EPOCH, self.loss_list_quantum[-1], self.accuracy_quantum_val))
        plt.ioff()
        plt.title("Quantum, {} p, {} lr, {} e, {} i".format(self.NCIRCUITS,self.LR,self.EPOCH,self.MOM))
        plt.plot(self.loss_list_quantum,'.')
        plt.grid()
        plt.savefig("Quantum{}p{}lr{}e{}i.jpg".format(self.NCIRCUITS,self.LR,self.EPOCH,self.MOM))
        plt.close()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total_loss = []
            for batch_idx, (data, target) in enumerate(self.testset):
                output = self.model(data)
                pred = output.argmax(dim = 1, keepdim = True)
                # print('output: ',output,'prediction: ',pred,'target: ',target)
                self.pq.append(pred)
                self.tq.append(target)
                correct += pred.eq(target.view_as(pred)).sum().item()
                self.correct=correct
                loss = self.loss(output, target)
                total_loss.append(loss.item())
            self.loss_quantum=sum(total_loss)/len(total_loss)
            self.accuracy_quantum=correct/len(self.testset)*100
            print('Quantum performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
                self.loss_quantum,
                self.accuracy_quantum)
                )
        confmat_q=torchmetrics.ConfusionMatrix(num_classes=2)
        self.pq=torch.tensor(self.pq)
        self.tq=torch.tensor(self.tq)
        self.tp_q=confmat_q(self.pq,self.tq)[0][0]/(confmat_q(self.pq,self.tq)[0][0]+confmat_q(self.pq,self.tq)[1][0])
        self.tn_q=confmat_q(self.pq,self.tq)[1][1]/(confmat_q(self.pq,self.tq)[0][1]+confmat_q(self.pq,self.tq)[1][1])

    def save(self,iteration):
        torch.save(self.model,"Models/Quantum__LR_{}__E_{}__M_{}__A_{:.1f}__{}.pt".format(self.LR,self.EPOCH,self.MOM,self.correct/len(self.testset)*100,iteration))
        np.savetxt("data/AccuracyListQ_{}p_{}e_{}.csv".format(self.NCIRCUITS,self.EPOCH,iteration),self.accuracy_val)
        np.savetxt("data/LossListQ_{}p_{}e_{}.csv".format(self.NCIRCUITS,self.EPOCH,iteration),self.loss_list_quantum)
        self.dataset=[self.NCIRCUITS,
        # self.conv,
        self.LR,
        self.EPOCH,
        # self.quantum_training_time,
        self.loss_quantum,
        self.tp_q,
        self.tn_q,
        self.accuracy_quantum,
        iteration
        ]
        return self.dataset

