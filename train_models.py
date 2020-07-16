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

# Data args
parser.add_argument('--data-dir', type=str, default='../../data', help='path to data directory')
parser.add_argument('--dataset', type=str, default='CIFAR', choices=['CIFAR'], help='the dataset to train on')
parser.add_argument('--batch-size-train', type=int, default=100, help='batch size for training')
parser.add_argument('--batch-size-test', type=int, default=100, help='batch size for testing')

# Training args
parser.add_argument('--save-acc', action='store_true', help='save per class accuracies of the model after each epoch')
parser.add_argument('--acc-save-path', type=str, default='models/accuracies.npz')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate for training')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--num-classes', type=int, default=100, help='number of classes to train/test on')

args = parser.parse_args()


def save_model(model_path):
    torch.save(model.state_dict(), model_path)


def test(model, loader, device=0):
    model.eval()
    num_classes = model.fc.out_features
    total, correct = np.zeros(num_classes), np.zeros(model.fc.out_features)
    class_idxs = np.arange(num_classes)[None].repeat(loader.batch_size, axis=0)
    for i, x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        # TODO debug
        y, pred = (y.cpu().numpy()[:, None] == class_idxs), (pred.cpu().numpy()[:, None] == class_idxs)
        total += y.sum(axis=0)
        correct += np.logical_and(pred, y).sum(axis=0)

    print('%d/%d (%.2f%%)' % (correct.sum(), total.sum(), correct.sum() / total.sum() * 100.))
    return correct, total


def train(model, train_loader, test_loader,
          epochs=100,
          lr=0.005,
          save_acc=False,
          acc_save_path=None,
          model_save_path='final-model.pth',
          device=0,
          seed=None):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    if seed:
        torch.manual_seed(seed)  # seed dataloader shuffling
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
            break
        print('Mean loss for epoch %d: %.4f' % (e, sum(losses) / len(losses)))
        print('Test accuracy for epoch %d:' % e, end=' ')
        correct_, total_ = test(model, test_loader, device=device)
        total += [total_]
        correct += [correct_]
        if save_acc:
            np.savez(acc_save_path, correct=np.stack(correct, axis=0), total=np.stack(total, axis=0))
        save_model(model_save_path)


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
    train(model, train_loader, test_loader,
          epochs=args.epoch,
          lr=args.lr,
          save_acc=args.save_acc,
          acc_save_path=args.acc_save_path,
          model_save_path=args.final_model_path,
          device=0,
          seed=args.seed)
