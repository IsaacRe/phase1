import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from utils.hook_management import HookManager
from utils.helpers import get_named_modules_from_network
from train_models import train, get_dataloaders, initialize_model, test, save_model
from argument_parsing import *


def sparse_grad_training(init_model_args: ModelInitArgs, train_args: SparseTrainingArgs,
                         train_loader: DataLoader, test_loader: DataLoader,
                         device=0):
    network = initialize_model(init_model_args, device=device)
    network.train()

    def get_optim(lr):
        return torch.optim.SGD(network.parameters(),
                               lr=lr,
                               nesterov=train_args.nesterov,
                               momentum=train_args.momentum,
                               weight_decay=train_args.weight_decay)

    lr = train_args.lr
    optim = get_optim(lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    torch.manual_seed(train_args.seed)  # seed dataloader shuffling

    for e in range(train_args.epochs):
        # check for lr decay
        if e in train_args.decay_epochs:
            lr /= train_args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, train_args.epochs))
        losses = []
        perc_grad_pruned = []

        for i, x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            out = network(x)
            loss = loss_fn(out, y)
            loss.backward()

            # threshold weight gradients
            total_grad_pruned = 0
            total_grad = 0
            for module in get_named_modules_from_network(network).values():
                with torch.no_grad():
                    abs_grad = module.weight.grad.abs()
                    mean_grad = abs_grad.mean()
                    max_grad = abs_grad.max()
                    threshold = (mean_grad * train_args.mean_max_coef + max_grad) / (train_args.mean_max_coef + 1)
                    grad_mask = abs_grad < threshold
                    module.weight.grad[grad_mask] = 0.0

                    total_grad += len(grad_mask.flatten())
                    total_grad_pruned += len(np.where(grad_mask.cpu())[0])

            perc_grad_pruned += [total_grad_pruned / total_grad]
            #print(perc_grad_pruned[-1])

            optim.step()
            optim.zero_grad()
            losses += [loss.item()]

        print('Mean loss for epoch %d: %.4f' % (e, sum(losses) / len(losses)))
        print('Average percent of gradients pruned: %.2f' % (sum(perc_grad_pruned) / len(perc_grad_pruned) * 100.))
        print('Test accuracy for epoch %d:' % e, end=' ')

        network.eval()
        correct_, total_ = test(network, test_loader, device=device)
        network.train()
        total += [total_]
        correct += [correct_]
        if train_args.save_acc:
            np.savez(train_args.acc_save_path, correct=np.stack(correct, axis=0), total=np.stack(total, axis=0))
        save_model(network, train_args.model_save_path, device=device)


if __name__ == '__main__':
    data_args, model_init_args, train_args = \
        parse_args(DataArgs, ModelInitArgs, SparseTrainingArgs)
    dataloaders = get_dataloaders(data_args)
    sparse_grad_training(model_init_args, train_args, *dataloaders, device=0)
