import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as col
import quanvolutional_filter as quanv
import QNN_Final as qnn

IMG_SIZE = 24
N_QUBITS_QUANV = 4
N_QUBITS_QNN = 4
N_EPOCHS = 15
N_CIRCUITS = 8

class Model(torch.nn.Module):

    def __init__(self, skip) -> None:
        super().__init__()
        self.skip = skip
        self.qf1 = quanv.TrainableQuanvolutionalFilter(N_QUBITS_QUANV, IMG_SIZE, pool = skip)
        self.k = np.sqrt(N_QUBITS_QUANV).astype(int)
        if skip:
            self.s = np.floor((IMG_SIZE)/self.k).astype(int)
        else:
            self.s = np.floor((IMG_SIZE-1)/self.k).astype(int)
        self.fc1 = torch.nn.Linear(N_QUBITS_QUANV*self.s*self.s, N_QUBITS_QNN*N_CIRCUITS)
        self.qc = qnn.TorchCircuit.apply
        self.out = torch.nn.Linear(N_CIRCUITS, 2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf1(x, use_qiskit = use_qiskit)
        if not self.skip:
            x = F.avg_pool2d(x, self.k)#.view(N_QUBITS, self.s, self.s)
        x = x.reshape(-1, N_QUBITS_QUANV*self.s*self.s)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = x.view(-1)
        x = torch.cat([(x[i * 2 : i*2 + 2]) / torch.norm(x[i * 2: i*2 + 2]) 
                       for i in range(N_CIRCUITS*2)], dim=-1) # normalize sin and cos for each angle
        x = torch.stack([torch.atan2(x[i * 2], x[i*2 + 1]) for i in range(N_CIRCUITS*2)]) # convert to angles
        x = torch.cat([self.qc(x[i * 2 : i*2 + 2]) for i in range(N_CIRCUITS)], dim=1) # QUANTUM LAYER
        x = self.out(x)

        return F.log_softmax(x,dim=1)
    
# Training subroutine
def train(dataloader, model, device, optimizer):
    target_all = []
    output_all = []
    for data, label in dataloader:
        inputs = data.to(device)
        targets = label.to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}', end = '\r')
        target_all.append(targets)
        output_all.append(outputs)
    target_all = torch.cat(target_all, dim = 0)
    output_all = torch.cat(output_all, dim = 0)
    _, indices = output_all.topk(1, dim = 1)
    masks = indices.eq(target_all.view(-1,1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    print('Training set accuracy: {}'.format(accuracy))

# Validation testing function
def valid_test(dataloader, split, model, device, qiskit = False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for data, label in dataloader:
            inputs = data.to(device)
            targets = label.to(device)
            
            outputs = model(inputs, use_qiskit = qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim = 0)
        output_all = torch.cat(output_all, dim = 0)

    _, indices = output_all.topk(1, dim = 1)
    masks = indices.eq(target_all.view(-1,1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    print(f'{split} set accuracy: {accuracy}')
    print(f'{split} set loss: {loss}')
    return accuracy, loss

def run(quanv_model, device, test_loader, train_loader, progress = False):
    model = quanv_model.to(device) #HybridModel().to(device)
    #model_without_qf = HybridModel_without_qf().to(device)
    n_epochs = N_EPOCHS
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.1, nesterov=True)
    #Quanv optimiser: optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    accu_list1 = []
    loss_list1 = []
    filter_set = []

    for epoch in tqdm(range(1, n_epochs+1)):
        # Train
        print(f'Epoch {epoch}:')
        train(train_loader, model, device, optimizer)
        print(optimizer.param_groups[0]['lr'])
        if progress:
            if epoch == 1 or epoch % 10 == 0:
                filters = model.qf1(x_train[12][0])
                filter_set.append(filters)
        # Validation test

        accu, loss = valid_test(test_loader, 'test', model, device, qiskit = False)
        accu_list1.append(accu)
        loss_list1.append(loss)
        scheduler.step()
    #torch.save(model, 'model3_v1.pt')
    if progress:
        return accu_list1, loss_list1, filter_set
    return accu_list1, loss_list1

if __name__ == '__main__':

    x_train = quanv.CustomImageDataset(
        annotations_file = 'brain_cancer_output/val.csv',
        img_dir = 'brain_cancer_output/val/Brain Tumor/',
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
    )

    x_test = quanv.CustomImageDataset(
        annotations_file = 'brain_cancer_output/test.csv',
        img_dir = 'brain_cancer_output/test/Brain Tumor/',
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
    )

    train_loader = torch.utils.data.DataLoader(x_train, batch_size = 1, shuffle = True) # 376 items
    test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True) # 753 items

    # Define device, models, optimizer, scheduler and number of epochs
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #c_acc, c_loss = run(ClassicalTrial(), device, test_loader, train_loader)
    q1_acc, q1_loss, filters = run(Model(skip = True), device, test_loader, train_loader, progress = True)
    #q2_acc, q2_loss = run(FinalModel(skip = False), device, test_loader, train_loader)
   
    # Plot filters over time
    f, axarr = plt.subplots(filters[0].size(0), len(filters))
    epoch_list = [1,10,20,30]
    for k, fil in enumerate(filters):
        norm = col.Normalize(vmin=0,vmax=1)
        for c in range(fil.size(0)):
            axarr[c,0].set_ylabel('Channel {}'.format(c))
            axarr[0,k].set_title('Epoch {}'.format(epoch_list[k]))
            if k != 0:
                axarr[c,k].yaxis.set_visible(False)
            img = fil[c,:].detach().numpy()
            axarr[c,k].imshow(img, norm=norm, cmap='gray')
    #plt.savefig('imgs3b.png')
    plt.show()
    plt.close()
    
    norm = col.Normalize(vmin=0,vmax=1)
    og_img = x_train[12][0].squeeze().detach().numpy()
    plt.imshow(og_img, norm = norm, cmap = 'gray')
    #plt.savefig('og_img33.png')
    plt.show()
    plt.close()
    
    # Accuracy plot
    #plt.plot(c_acc, label = 'Classical')
    plt.plot(q1_acc, label = 'Quanv with stride')
    #plt.plot(q2_acc, label = 'Quanv without stride')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig('acc3b_plot.png')
    plt.show()
    plt.close()

    # Loss plot
    #plt.plot(c_loss, label = 'Classical')
    plt.plot(q1_loss, label = 'Quantum')
    #plt.plot(q2_loss, label = 'Quanv without stride')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('loss3b_plot.png')
    plt.show()
    
    #print('Classical: ', c_acc[-1])
    print('Quanv with stride: ', q1_acc[-1])