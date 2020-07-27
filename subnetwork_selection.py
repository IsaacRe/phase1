from argument_parsing import *
import numpy as np
import torch.nn
import torch.optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.model_pruning import ModulePruner, PruneProtocol
from train_models import test, train, get_dataloaders, model_factories


def load_model(architecture, model_path, device=0):
    state_dict = torch.load(model_path)
    num_class = state_dict['fc.weight'].shape[0]
    model = model_factories[architecture]()
    model.fc = torch.nn.Linear(model.fc.in_features, num_class, bias=True)
    model.load_state_dict(state_dict)
    model.cuda(device)
    return model


def retrain_bn(model, train_loader: DataLoader, device=0):
    model.train()
    reset_bn(model)
    bn_params = []
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_params += list(m.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(bn_params, lr=0.1, momentum=0.9, nesterov=True)
    for i, x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        optim.zero_grad()


def reset_bn(model):
    def reset_bn_weights(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_parameters()
    model.apply(reset_bn_weights)


def compute_mask_accuracy(masks, ref_masks):
    ret_dict = {}
    total_size = 0
    total_negative_ref = 0
    total_overlap = 0
    total_negative_overlap = 0
    total_positive_ref = 0
    total_positive_overlap = 0
    for module_name in masks:
        mask, ref_mask = masks[module_name], ref_masks[module_name]

        # get number of points, correct, negatives, and true negatives
        mask_size = len(mask.flatten())
        overlap = (mask == ref_mask).sum().item()
        negative_ref = (~ref_mask).sum().item()
        negative_overlap = np.logical_and(~mask, ~ref_mask).sum().item()
        positive_ref = ref_mask.sum().item()
        positive_overlap = np.logical_and(mask, ref_mask).sum().item()

        # total counters
        total_size += mask_size
        total_overlap += overlap
        total_negative_ref += negative_ref
        total_negative_overlap += negative_overlap
        total_positive_ref += positive_ref
        total_positive_overlap += positive_overlap

        # compute per-module accuracy and recall
        accuracy = overlap / mask_size  # total mask overlap
        retained_recall = negative_overlap / negative_ref  # overlap of non-masked values in reference mask
        pruned_recall = positive_overlap / positive_ref  # overlap of masked values in reference mask
        ret_dict['%s_accuracy' % module_name] = accuracy * 100.
        ret_dict['%s_retained_recall' % module_name] = retained_recall * 100.
        ret_dict['%s_pruned_recall' % module_name] = pruned_recall * 100.

    ret_dict['mean_accuracy'] = total_overlap / total_size * 100.
    ret_dict['mean_retained_recall'] = total_negative_overlap / total_negative_ref * 100.
    ret_dict['mean_pruned_recall'] = total_positive_overlap / total_positive_ref * 100.

    return ret_dict


def get_pruners(*protocols: PruneProtocol, device=0, networks=None, dataloaders=None):
    if networks:
        assert len(networks) == len(protocols)
    else:
        networks = [None] * len(protocols)
    if dataloaders:
        assert len(dataloaders) == len(protocols)
    else:
        dataloaders = [None] * len(protocols)
    return (ModulePruner(protocol,
                         device=device,
                         network=network,
                         dataloader=dataloader)
            for protocol, network, dataloader in zip(protocols, networks, dataloaders))


def subnetwork_experiments(args: ExperimentArgs,
                           init_protocol: PruneProtocol, final_protocol: PruneProtocol,
                           train_loader: DataLoader, test_loader: DataLoader,
                           device=0):
    # load final model
    final_model = load_model(args.arch, args.final_model_path, device=device)
    final_model.eval()
    print('Loaded final model from %s' % args.final_model_path)

    # load initial model
    init_model = load_model(args.arch, args.init_model_path, device=device)
    print('Loaded initialized model from %s' % args.init_model_path)

    # test final model accuracy before pruning
    """
    print('Testing final model accuracy before pruning')
    correct, total = test(final_model, test_loader, device=device)
    acc_no_prune = correct.sum() / total.sum() * 100.
    print('Model accuracy before pruning: %.2f' % acc_no_prune)
    """
    acc_no_prune = 66.76

    # get pruners
    final_pruner, init_pruner = get_pruners(final_protocol, init_protocol,
                                            device=device,
                                            networks=(final_model, init_model))

    # compute reference prune masks
    final_pruner = ModulePruner(final_protocol,
                                device=device,
                                network=final_model)

    print('Computing prune masks for final model...')
    final_masks = final_pruner.compute_prune_masks(reset=False)

    # test final model performance when final prune mask is used
    print('Testing final model performance after pruning from final model...')
    with final_pruner.prune(clear_on_exit=True):
        retrain_bn(final_model, train_loader, device=device)
        final_model.eval()
        correct, total = test(final_model, test_loader, device=device)
    final_acc = correct.sum() / total.sum() * 100.
    print('Model accuracy using pruning on final model: %.2f' % final_acc)

    # compute initial prune masks
    print('Computing prune masks for initialized model...')
    init_masks = init_pruner.compute_prune_masks(reset=not args.retrain)

    # test final model performance when initial prune mask is used
    print('Testing final model performance after pruning from model at initialization...')
    final_pruner.set_prune_masks(**init_masks)
    with final_pruner.prune(clear_on_exit=False):
        retrain_bn(final_model, train_loader, device=device)
        final_model.eval()
        correct, total = test(final_model, test_loader, device=device)
    init_acc = correct.sum() / total.sum() * 100.
    print('Model accuracy using pruning at model initialization: %.2f' % init_acc)

    # compute overlap between prune masks
    print('Computing overlap between prune masks of initialized model and final model...')
    mask_accuracy_dict = compute_mask_accuracy(init_masks, final_masks)
    print('Mean mask accuracy: %.2f' % mask_accuracy_dict['mean_accuracy'])
    print('Mean mask retained recall: %.2f' % mask_accuracy_dict['mean_retained_recall'])
    print('Mean mask pruned recall: %.2f' % mask_accuracy_dict['mean_pruned_recall'])

    # test final model performance with random prune mask
    print('Testing final model performance after random pruning...')

    def make_random_mask(mask):
        neg_mask = torch.zeros_like(mask).type(torch.bool)
        flat_mask = neg_mask.flatten()
        length = flat_mask.shape[0]
        flat_mask[np.random.choice(length, int(length * init_protocol.prune_ratio), replace=False)] = True
        return neg_mask

    # test final model with random pruning
    """
    random_masks = {name: make_random_mask(mask) for name, mask in init_masks.items()}
    final_pruner.set_prune_masks(**random_masks)
    with final_pruner.prune(clear_on_exit=True):
        correct, total = test(final_model, test_loader, device=device)
    random_acc = correct.sum() / total.sum() * 100.
    print('Model accuracy using random pruning: %.2f' % random_acc)
    """
    random_acc = None

    if args.save_results:
        print('Saving experiment results to %s' % args.results_filepath)
        np.savez(args.results_filepath,
                 final_accuracy=acc_no_prune,
                 init_subnet_accuracy=init_acc,
                 final_subnet_accuracy=final_acc,
                 random_subnet_accuracy=random_acc,
                 **mask_accuracy_dict)


def activation_pruning_experiments(args: ExperimentArgs, protocol: PruneProtocol,
                                   train_loader: DataLoader, test_loader: DataLoader,
                                   device=0):
    protocol.prune_by = 'online'
    model = load_model(args.arch, args.final_model_path, device=device)
    model.eval()

    print('Testing final model accuracy before pruning')
    #correct, total = test(model, test_loader, device=device)
    #acc_no_prune = correct.sum() / total.sum() * 100.
    #print('Model accuracy before pruning: %.2f' % acc_no_prune)

    pruner = ModulePruner(protocol,
                          device=device,
                          dataloader=test_loader,
                          network=model)

    print('Testing final model accuracy with real-time activation pruning')
    with pruner.prune(clear_on_exit=True):
        retrain_bn(model, train_loader, device=device)
        model.eval()
        correct, total = test(model, test_loader, device=device)
    prune_acc = correct.sum() / total.sum() * 100.
    print('Model accuracy with pruning: %.2f' % prune_acc)


def retrain_experiments(args: ExperimentArgs, train_args: RetrainingArgs,
                        final_protocol: PruneProtocol, init_protocol: PruneProtocol,
                        train_loader: DataLoader, test_loader: DataLoader,
                        use_final_subnetwork=True,
                        device=0):
    # load init model
    init_model = load_model(args.arch, args.init_model_path, device=device)
    init_model.eval()
    print('Loaded initialized model from %s' % args.init_model_path)

    (init_pruner,) = get_pruners(init_protocol,
                                 device=device,
                                 networks=(init_model,))

    if use_final_subnetwork:
        # load final model
        final_model = load_model(args.arch, args.final_model_path, device=device)
        print('Loaded final model from %s' % args.final_model_path)

        (final_pruner,) = get_pruners(final_protocol,
                                      device=device,
                                      networks=(final_model,))

        init_pruner.set_prune_masks(**final_pruner.compute_prune_masks(reset=True))
        del final_model

    with init_pruner.prune(clear_on_exit=True, recompute_masks=False):
        print('Retraining pruned subnetwork of model at initialization...')
        train(train_args, init_model, train_loader, test_loader, device=device)


if __name__ == '__main__':
    args, prune_init_args, prune_final_args, data_args, train_args = \
        parse_args(ExperimentArgs, PruneInitArgs, PruneFinalArgs, DataArgs, RetrainingArgs)
    init_protocol = PruneProtocol(namespace=prune_init_args)
    final_protocol = PruneProtocol(namespace=prune_final_args)
    dataloaders = get_dataloaders(data_args)

    # Run experiments
    #activation_pruning_experiments(args, final_protocol, *dataloaders, device=0)
    #subnetwork_experiments(args, init_protocol, final_protocol, *dataloaders, device=0)
    retrain_experiments(args, train_args, init_protocol, final_protocol, *dataloaders,
                        use_final_subnetwork=args.use_final_subnetwork, device=0)
