from argument_parsing import *
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
from torchvision.models import resnet18, resnet34
from dataset import get_dataloader_cifar


model_factories = {
    'resnet18': resnet18,
    'resnet34': resnet34
}


def save_model(model, save_path, device=0):
    model.cpu()
    torch.save(model.state_dict(), save_path)
    if device != 'cpu':
        model.cuda(device)


def test(model, loader, device=0):
    with torch.no_grad():
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


def train(args: TrainingArgs, model, train_loader, test_loader, device=0):
    model.train()
    def get_optim(lr):
        return torch.optim.SGD(model.parameters(),
                               lr=lr,
                               nesterov=args.nesterov,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    lr = args.lr
    optim = get_optim(lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    torch.manual_seed(args.seed)  # seed dataloader shuffling

    for e in range(args.epochs):
        # check for lr decay
        if e in args.decay_epochs:
            lr /= args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, args.epochs))
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

        model.eval()
        correct_, total_ = test(model, test_loader, device=device)
        model.train()
        total += [total_]
        correct += [correct_]
        if args.save_acc:
            np.savez(args.acc_save_path, correct=np.stack(correct, axis=0), total=np.stack(total, axis=0))
        save_model(model, args.model_save_path, device=device)


def initialize_model(args: ModelInitArgs, device=0):
    # load network
    torch.manual_seed(args.seed)  # seed random network initialization
    model = model_factories[args.arch](pretrained=args.pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes, bias=True)
    if device != 'cpu':
        model.cuda(device)
    return model


# load test/train dataloaders
def get_dataloaders(args: DataArgs):
    if args.dataset == 'CIFAR':
        train_loader = get_dataloader_cifar(args.batch_size_train,
                                            data_dir=args.data_dir,
                                            num_classes=args.num_classes,
                                            train=True,
                                            num_workers=args.num_workers)
        test_loader = get_dataloader_cifar(args.batch_size_test,
                                           data_dir=args.data_dir,
                                           num_classes=args.num_classes,
                                           train=False,
                                           num_workers=args.num_workers)
    return train_loader, test_loader


if __name__ == '__main__':
    model_args, save_init_args, data_args, train_args = parse_args(ModelInitArgs, InitModelPath, DataArgs, TrainRefModelArgs)
    model = initialize_model(model_args, device=0)
    save_model(model, save_init_args.init_model_path)
    train_loader, test_loader = get_dataloaders(data_args)
    train(train_args, model, train_loader, test_loader, device=0)
