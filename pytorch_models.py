

import torch
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# import torchmetrics

class Custom_Model:
    def __init__(self, model, model_name:str, optimizer=None, loss_function=None):
        print(f'Init custom model...')

        self.model = model
        self.model_name = model_name

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        else:
            self.optimizer = optimizer

        if loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            self.loss_function = loss_function

        self.writer = None

    def train_one_epoch(self, train_loader, epoch_index):
        running_loss = 0.0
        last_loss = 0.0
        correct = 0
        total = 0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        print(f'-- Batches: {len(train_loader)}')
        for i, data in enumerate(tqdm(train_loader)):
            # Every data instance is an input + label pair
            inputs, labels = data
            # print(i, data, inputs, labels)

            # Zero your gradients for every batch!
            # is this needed??
            # self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # compute accuracy
            pred = outputs.argmax(1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.shape[0] * labels.shape[1]
            running_acc = correct/total

            # Compute the loss and its gradients
            loss = self.loss_function(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print(f'  batch {i+1} loss: {last_loss} acc: {running_acc}')
                tb_x = epoch_index * len(train_loader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        total_accuracy = correct/total
        return last_loss, total_accuracy

    def train(self, num_epochs, train_loader, valid_loader):
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'./runs/{self.model_name}_{timestamp}')
        # epoch_number = 0
        EPOCHS = num_epochs

        best_valid_loss = 1_000_000
        last_valid_loss = 1000000

        # total_acc = torchmetrics.Accuracy()
        print('Early stopping conditioned on validation loss')
        for epoch in range(EPOCHS):
            # print(f'epoch {epoch}')
            print(f'--------------------')
            print(f'epoch number: {epoch + 1}')

            # turn model to training
            self.model.train(True)
            avg_loss, avg_acc = self.train_one_epoch(train_loader, epoch)

            running_validation_loss = 0.0
            valid_correct = 0
            valid_total = 0

            # set model to eval mode to evaluate epoch
            self.model.eval()
            # run model on validation set
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs)
                vloss = self.loss_function(voutputs, vlabels)
                running_validation_loss += vloss

                # compute accuracy
                vpred = voutputs.argmax(1, keepdim=True)
                valid_correct += vpred.eq(vlabels.view_as(vpred)).sum().item()
                valid_total += vlabels.shape[0] * vlabels.shape[1]

            valid_acc = valid_correct/valid_total
            avg_vloss = running_validation_loss / (i+1)
            print(f'LOSS: train - {avg_loss} valid - {avg_vloss}\n')
            print(f'ACC: train - {avg_acc} valid  -{valid_acc}\n')
            print(f'--------------------')

            # log running loss averaged per batch
            # both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                            { "Training" : avg_loss, 'Validation' : avg_vloss },
                            epoch + 1)
            self.writer.flush()

            # track best performance and save the model's state
            if avg_vloss < best_valid_loss:
                best_valid_loss = avg_vloss
                model_path = f'./models/{self.model_name}_{timestamp}_{epoch}'
                torch.save(self.model.state_dict(), model_path)

            # early stopping
            if avg_vloss > last_valid_loss:
                print('validation loss stopped decreasing')
                break
            else:
                last_valid_loss = avg_vloss

    def evaluate_model(self, test_loader):
        self.model.eval()
        acc = 0
        total = 0
        for i, data in enumerate(tqdm(test_loader)):

            points, labels = data

            # perform prediction
            outputs = self.model(points)

            # get index of max log probability
            pred = outputs.argmax(1, keepdim=True)

            # count the total correctly predicted vertices
            correct = pred.eq(labels.view_as(pred)).sum().item()
            acc += correct

            # count total vertices
            total += labels.shape[0] * labels.shape[1]

        total_accuracy = acc / total
        # print('Test Accuracy', total_accuracy)
        return total_accuracy

    def predict(self, points):
        # takes input of numpy array and outputs numpy array
        if len(points.shape) == 3:
            points = torch.tensor(points)
        else:
            points = torch.tensor(points).unsqueeze(0)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(points)
            pred = outputs.argmax(1, keepdim=True)

        pred = pred.squeeze(0).squeeze(0).detach().numpy()
        return pred # return numpy list of vertex labels