from argparse import ArgumentParser
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
from torchvision.models import resnet34
from dataset import get_dataloader_cifar


parser = ArgumentParser()

parser.add_argument('--seed', type=int, default=1, help='seed for random number generators')
parser.add_argument('--init-model-path', type=str, help='path to save file of the initialized model before training')
parser.add_argument('--final-model-path', type=str, help='path to save file of the final model')
parser.add_argument('--pretrained', action='store_true', help='start from pretrained initialization')
parser.add_argument('--data-dir', type=str, default='../../data', help='path to data directory')
parser.add_argument('--save-acc', action='store_true', help='save per class accuracies of the model after each epoch')
parser.add_argument('--acc-save-path', type=str, default='models/accuracies.npz')

# Training args
parser.add_argument('--dataset', type=str, default='CIFAR', help='the dataset to train on')
parser.add_argument('--batch-size-train', type=int, default=100, help='batch size for training')
parser.add_argument('--batch-size-test', type=int, default=100, help='batch size for testing')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate for training')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--num-classes', type=int, default=100, help='number of classes to train/test on')

args = parser.parse_args()


def save_model(model_path):
    torch.save(model.state_dict(), model_path)


def test(device=0):
    model.eval()
    total, correct = np.zeros(args.num_classes), np.zeros(args.num_classes)
    class_idxs = np.arange(args.num_classes)[None].repeat(args.batch_size_test, axis=0)
    for i, x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        y, pred = (y.cpu().numpy()[:, None] == class_idxs), (pred.cpu().numpy()[:, None] == class_idxs)
        total += y.sum(axis=0)
        correct += np.logical_and(pred, y).sum(axis=0)

    print('%d/%d (%.2f%%)' % (correct.sum(), total.sum(), correct.sum() / total.sum() * 100.))
    return correct, total


def train(epochs, device=0):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    torch.manual_seed(args.seed)  # seed dataloader shuffling
    for e in range(epochs):
        print('Beginning epoch %d/%d' % (e + 1, epochs))
        losses = []
        for i, x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses += [loss.item()]
        print('Mean loss for epoch %d: %.4f' % (e, sum(losses) / len(losses)))
        print('Test accuracy for epoch %d:' % e, end=' ')
        total_, correct_ = test(device=device)
        total += [total_]
        correct += [correct_]
        if args.save_acc:
            np.savez(args.acc_save_path, correct=np.stack(correct, axis=0), total=np.stack(total, axis=0))
        save_model(args.final_model_path)


# load network
torch.manual_seed(args.seed)  # seed random network initialization
model = resnet34(pretrained=args.pretrained)
model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes, bias=True)
model.cuda()
save_model(args.init_model_path)

# load test/train dataloaders
if args.dataset == 'CIFAR':
    train_loader = get_dataloader_cifar(args.batch_size_train,
                                        data_dir=args.data_dir,
                                        num_classes=args.num_classes,
                                        train=True)
    test_loader = get_dataloader_cifar(args.batch_size_test,
                                       data_dir=args.data_dir,
                                       num_classes=args.num_classes,
                                       train=False)

if __name__ == '__main__':
    train(args.epoch, device=0)
