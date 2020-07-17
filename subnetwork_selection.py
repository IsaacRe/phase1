from argument_parsing import *
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from utils.model_pruning import ModulePruner, PruneProtocol
from train_models import test, train, get_dataloaders


def load_model(args: NumClass, model_path, device=0):
    model = resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes, bias=True)
    model.load_state_dict(torch.load(model_path))
    model.cuda(device)
    return model


def compute_mask_accuracy(masks, ref_masks):
    ret_dict = {}
    total_size = 0
    total_negative_ref = 0
    total_overlap = 0
    total_negative_overlap = 0
    for module_name in masks:
        mask, ref_mask = masks[module_name], ref_masks[module_name]

        # get number of points, correct, negatives, and true negatives
        mask_size = len(mask.flatten())
        overlap = (mask == ref_mask).sum()
        negative_ref = (~ref_mask).sum()
        negative_overlap = np.logical_and(~mask, ~ref_mask).sum()

        # total counters
        total_size += mask_size
        total_overlap += overlap
        total_negative_ref += negative_ref
        total_negative_overlap += negative_overlap

        # compute per-module accuracy and recall
        accuracy = overlap / mask_size  # total mask overlap
        recall = negative_overlap / negative_ref  # overlap of non-masked values in reference mask
        ret_dict['%s_accuracy' % module_name] = accuracy * 100.
        ret_dict['%s_recall' % module_name] = recall * 100.

    ret_dict['mean_accuracy'] = total_overlap / total_size * 100.
    ret_dict['mean_recall'] = total_negative_overlap / total_negative_ref * 100.

    return ret_dict


# TODO
def subnetwork_experiments(args: SubnetworkSelectionArgs, train_args: RetrainingArgs,
                           init_protocol: PruneProtocol, final_protocol: PruneProtocol,
                           train_loader: DataLoader, test_loader: DataLoader,
                           device=0):
    # load final model
    final_model = load_model(args, args.final_model_path, device=device)
    print('Loaded final model from %s' % args.final_model_path)

    # test final model accuracy before pruning
    print('Testing final model accuracy before pruning')
    #correct, total = test(final_model, test_loader, device=device)
    #acc_no_prune = correct.sum() / total.sum() * 100.
    #print('Model accuracy before pruning: %.2f' % acc_no_prune)

    # compute reference prune masks
    final_pruner = ModulePruner(final_protocol,
                                device=device,
                                network=final_model)

    print('Computing prune masks for final model...')
    final_masks = final_pruner.compute_prune_masks(reset=False)

    # test final model performance when final prune mask is used
    print('Testing final model performance after pruning from final model...')
    with final_pruner.prune(clear_on_exit=True):
        correct, total = test(final_model, test_loader, device=device)
    final_acc = correct.sum() / total.sum() * 100.
    print('Model accuracy using pruning on final model: %.2f' % final_acc)

    # load initial model and compute prune masks
    init_model = load_model(args, args.init_model_path, device=device)
    print('Loaded initialized model from %s' % args.init_model_path)
    init_pruner = ModulePruner(init_protocol,
                               device=device,
                               network=init_model)
    print('Computing prune masks for initialized model...')
    init_masks = init_pruner.compute_prune_masks(reset=not args.retrain)

    # compute overlap between prune masks
    print('Computing overlap between prune masks of initialized model and final model...')
    mask_accuracy_dict = compute_mask_accuracy(init_masks, final_masks)
    print('Mean mask accuracy: %.2f' % mask_accuracy_dict['mean_accuracy'])
    print('Mean mask recall: %.2f' % mask_accuracy_dict['mean_recall'])

    # test final model performance when initial prune mask is used
    print('Testing final model performance after pruning from model at initialization...')
    final_pruner.set_prune_masks(**init_masks)
    with final_pruner.prune(clear_on_exit=True):
        correct, total = test(final_model, test_loader, device=device)
    init_acc = correct.sum() / total.sum() * 100.
    print('Model accuracy using pruning at model initialization: %.2f' % init_acc)

    retrain_acc = None
    if args.retrain:
        with final_pruner.prune(clear_on_exit=True):
            print('Retraining pruned subnetwork of model at initialization...')
            train(train_args, init_model, train_loader, test_loader, device=device)
            print('Testing retrained subnetwork performance...')
            correct, total = test(init_model, test_loader, device=device)
            retrain_acc = correct.sum() / total.sum() * 100.
            print('Retrained subnetwork accuracy: %.2f' % retrain_acc)

    if args.save_results:
        print('Saving experiment results to %s' % args.results_filepath)
        np.savez(args.results_filepath,
                 final_accuracy=acc_no_prune,
                 init_subnet_accuracy=init_acc,
                 final_subnet_accuracy=final_acc,
                 retrain_subnet_accuracy=retrain_acc,
                 **mask_accuracy_dict)


"""

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

"""


if __name__ == '__main__':
    args, prune_init_args, prune_final_args, data_args, train_args = \
        parse_args(SubnetworkSelectionArgs, PruneInitArgs, PruneFinalArgs, DataArgs, RetrainingArgs)
    init_protocol = PruneProtocol(namespace=prune_init_args)
    final_protocol = PruneProtocol(namespace=prune_final_args)
    dataloaders = get_dataloaders(data_args)
    subnetwork_experiments(args, train_args, init_protocol, final_protocol, *dataloaders, device=0)
