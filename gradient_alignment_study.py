import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from utils.helpers import find_network_modules_by_name
from train_models import train, get_dataloaders, initialize_model, test
from argument_parsing import *


def grad_alignment_study(init_model_args: ModelInitArgs, train_args: TrainingArgs,
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
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    total, correct = [], []
    torch.manual_seed(train_args.seed)  # seed dataloader shuffling

    for e in range(train_args.epochs):
        # check for lr decay
        if e in train_args.decay_epochs:
            lr /= train_args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, train_args.epochs))
        losses = []

        for idx, (i, x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            out = network(x)
            loss = loss_fn(out, y)

            if e in train_args.test_epochs and idx == 0:
                # explore sample gradients
                print('\nComputing gradient alignment across network modules...')
                modules = find_network_modules_by_name(network, train_args.test_layers)
                grads = {n: [] for n in train_args.test_layers}
                for i in range(100):
                    loss[i].backward(retain_graph=True)

                    for n, m in zip(train_args.test_layers, modules):
                        grads[n] += [m.weight.grad.cpu()]

                    optim.zero_grad()

                mean = {}
                magnitude = {}
                variance = {}
                for n, grad in grads.items():
                    mean[n] = torch.stack(grad).mean(dim=0).numpy()
                    magnitude[n] = torch.stack(grad).abs().mean(dim=0).numpy()
                    variance[n] = ((torch.stack(grad).numpy() - mean[n][None]) ** 2).sum(axis=0)

                np.savez('gradient-study/alignment/metrics-epoch_%d.npz' % e,
                         **{n: [np.array(('mean', mean[n]), dtype=np.object),
                                np.array(('magnitude', magnitude[n]), dtype=np.object),
                                np.array(('variance', variance[n]), dtype=np.object)]
                            for n in train_args.test_layers})

            loss.mean().backward()

            optim.step()
            optim.zero_grad()
            losses += [loss.mean().item()]

        print('Mean loss for epoch %d: %.4f' % (e, sum(losses) / len(losses)))
        print('Test accuracy for epoch %d:' % e, end=' ')

        network.eval()
        correct_, total_ = test(network, test_loader, device=device)
        network.train()
        total += [total_]
        correct += [correct_]
        if train_args.save_acc:
            np.savez(train_args.acc_save_path, correct=np.stack(correct, axis=0), total=np.stack(total, axis=0))


if __name__ == '__main__':
    data_args, model_init_args, train_args = \
        parse_args(DataArgs, ModelInitArgs, GradAlignmentArgs)
    dataloaders = get_dataloaders(data_args)
    grad_alignment_study(model_init_args, train_args, *dataloaders, device=0)

