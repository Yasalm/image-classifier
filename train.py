import torch
from torch import nn
from torch import optim

from workspace_utils import active_session
import argparse

from helper import build_network
from utils import get_trainloader, get_valloader, get_testloader, get_accuracy, save_checkpoint


parser = argparse.ArgumentParser(description="Parser of training script")

# parser.add_argument ('data_dir', help = 'Provide data directory.', type = str)
parser.add_argument(
    '--save_dir', help='Provide saving directory. Optional argument, please include .pth', type=str)
parser.add_argument(
    '--arch', help='densenet121 can be used if this argument specified, otherwise resnet50 will be used', type=str)
parser.add_argument('--lr', help='Learning rate, default = 0.003', type=float)
parser.add_argument(
    '--hidden_units', help='Hidden units in Classifier. Default = 512', type=int)
parser.add_argument('--epochs', help='Number of epochs, default = 3', type=int)
parser.add_argument('--gpu', help="Option to use GPU", type=str)
parser.add_argument(
    '--validate_at', help="print score on vaildation at x forward passes, default 40", type=int)
# parser.add_argument ('--modelname', help = "Model name -- default = densenet121, options -- densenet121 - resnet50 ", type = str)

args = parser.parse_args()

if args.save_dir:
    save_dir = args.save_dir
else:
    save_dir = 'checkpoint.pth'

device = torch.device('cuda' if args.gpu else 'cpu')

if args.arch:
    model = build_network(
        'resnet50', hidden_layer1_units=args.hidden_units if args.hidden_units else 512)
else:
    model = build_network(
        hidden_layer1_units=args.hidden_units if args.hidden_units else 512)


# Load the datasets with ImageFolder
trainloader, train_dataset = get_trainloader()
valloader, _ = get_valloader()
testloader, _ = get_testloader()

# Using the image datasets and the trainforms, define the dataloaders
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


def train(epochs=3, lr=0.003, print_scores_in=40, gpu=True):

    # criterion
    criterion = nn.NLLLoss()
    # optimizer, we will only use classifier param. as features param are frozen.
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    # moving model
    model.to(device)
    if args.epochs:
        epochs = args.epochs

    if args.lr:
        lr = args.lr
    if args.validate_at:
        print_scores_in = args.validate_at
    steps = 0
    running_loss = 0

    with active_session():
        for e in range(epochs):
            print('Model Training...')
            for images, labels in trainloader:

                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_scores_in == 0:

                    test_loss, accuracy = 0, 0
                    # turn off dropput
                    model.eval()
                    with torch.no_grad():
                        for images, labels in valloader:
                            images, labels = images.to(
                                device), labels.to(device)
                            log_ps = model.forward(images)
                            test_loss += criterion(log_ps, labels)

                            # calculate accuracy
                            ps = torch.exp(log_ps)
                            _, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    print('Epoch: {}/{}.'.format(e+1, epochs),
                            'Train loss: {:.3f}.'.format(
                                running_loss/print_scores_in),
                            'Validation Loss: {:.3f}. '.format(
                                test_loss/len(valloader)),
                            'Validation Accuracy: {:.3f}'.format(accuracy/len(valloader)))
                    running_loss = 0
                    model.train()

    print('------------\nmodel trained')
    try:
        save_checkpoint(model, save_dir, train_dataset)
        print('Model Saved')
    except:
        print('Failed to save model')


if __name__ == "__main__":
    train()
