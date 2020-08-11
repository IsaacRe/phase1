from argparse import *
from typing import *


def parse_args(*arg_classes):
    _check_args()
    for ArgClass in arg_classes:
        parser = CustomParser(ArgClass)
        yield parser.parse_defined_args()


def get_all_bases(Class, stop_class=object):
    bases = []
    for base in Class.__bases__:
        bases += [base]
        if base == stop_class:
            continue
        bases += get_all_bases(base, stop_class=stop_class)
    return bases


def _check_args():
    # attempt to parse all provided arguments
    all_args = {}
    argnames = set()
    args = set()
    for ArgClass in all_argsets:
        for name, arg in ArgClass.ARGS.items():
            while name in argnames:
                name += ' '
            if arg in args:
                continue
            all_args[name] = arg
            argnames = argnames.union({name})
            args = args.union({arg})

    class AllArgs(ArgumentClass):
        ARGS = all_args

    CustomParser(AllArgs, allow_overlap=True).parse_args()


class Argument:

    def __init__(self, *flags: Text, **kwargs):
        self.flags = flags
        self.kwargs = kwargs

    def __repr__(self):
        return "Argument(%s, %s)" % (', '.join(self.flags),
                                     ', '.join(['%s=%s' % (k, str(v)) for k, v in self.kwargs.items()]))


class CustomParser(ArgumentParser):

    def __init__(self, ArgClass, *args, allow_overlap=False, **kwargs):
        super(CustomParser, self).__init__(*args, **kwargs)
        self.ARGUMENT_CLASS = ArgClass
        self._add_class_args()

    def _add_class_args(self):
        for ArgClass in set([self.ARGUMENT_CLASS] + get_all_bases(self.ARGUMENT_CLASS, stop_class=ArgumentClass)):
            for name, arg in ArgClass.ARGS.items():
                self.add_argument(*arg.flags, dest=name, **arg.kwargs)

    def parse_defined_args(self) -> Namespace:
        namespace, unknown = super(CustomParser, self).parse_known_args()
        return self.ARGUMENT_CLASS(namespace)


class ArgumentClass(Namespace):

    ARGS = {}

    def __init__(self, namespace):
        super(ArgumentClass, self).__init__()
        for k, v in namespace.__dict__.items():
            setattr(self, k, v)


# Define argument classes below
# Shared Arguments:


class NumClass(ArgumentClass):
    ARGS = {
        'num_classes':
            Argument('--num-classes', type=int, default=100, help='number of classes to train/test on')
    }


class Seed(ArgumentClass):
    ARGS = {
        'seed':
            Argument('--seed', type=int, default=1, help='seed for random number generators')
    }


class Architecture(ArgumentClass):
    ARGS = {
        'arch':
            Argument('--arch', type=str, default='resnet34', choices=['resnet18', 'resnet34'],
                     help='model architecture to use')
    }


class DataArgs(NumClass):
    ARGS = {
        'data_dir':
            Argument('--data-dir', type=str, default='../../data', help='path to data directory'),
        'dataset':
            Argument('--dataset', type=str, default='CIFAR', choices=['CIFAR'], help='the dataset to train on'),
        'batch_size_train':
            Argument('--batch-size-train', type=int, default=100, help='batch size for training'),
        'batch_size_test':
            Argument('--batch-size-test', type=int, default=100, help='batch size for testing'),
        'num_workers':
            Argument('--num-workers', type=int, default=4, help='number of dataloader workers')
    }


class FeatureDataArgs(DataArgs):
    ARGS = {
        'layer':
            Argument('--layer', type=str, default=None, help='layer at which to extract features for dataset'),
        'load_features_train':
            Argument('--load-features-train', type=str, default=None, help='path to trainset feature data to load'),
        'save_features_train':
            Argument('--save-features-train', type=str, default=None, help='path to save trainset feature data to'),
        'load_features_test':
            Argument('--load-features-test', type=str, default=None, help='path to testset feature data to load'),
        'save_features_test':
            Argument('--save-features-test', type=str, default=None, help='path to save testset feature data to')
    }


class InitModelPath(ArgumentClass):
    ARGS = {
        'init_model_path':
            Argument('--init-model-path', type=str,
                     help='path to save file of the initialized model before training')
    }


class FinalModelPath(ArgumentClass):
    ARGS = {
        'final_model_path':
            Argument('--final-model-path', type=str, help='path to save file of the final model')
    }


class TrainingArgs(Seed):
    ARGS = {
        'save_acc':
            Argument('--save-acc', action='store_true',
                     help='save per class accuracies of the model after each epoch'),
        'acc_save_path':
            Argument('--acc-save-path', type=str, default='models/accuracies.npz'),
        'lr':
            Argument('--lr', type=float, default=0.005, help='initial learning rate for training'),
        'lr_decay':
            Argument('--lrd', type=float, default=5, help='divisor for learning rate decay'),
        'decay_epochs':
            Argument('--decay-epochs', type=int, nargs='*', default=[60, 120, 160],
                     help='specify epochs during which to decay learning rate'),
        'nesterov':
            Argument('--nesterov', type=bool, default=True, help='whether to use nesterov optimization'),
        'momentum':
            Argument('--momentum', type=float, default=0.9, help='momentum for optimization'),
        'weight_decay':
            Argument('--weight-decay', type=float, default=5e-4, help='weight decay for optimization'),
        'epochs':
            Argument('--epochs', type=int, default=200, help='number of epochs to train for'),
    }


# Arguments exclusively for train_models.py:


class ModelInitArgs(Seed, NumClass, InitModelPath, Architecture):
    ARGS = {
        'pretrained':
            Argument('--pretrained', action='store_true', help='start from pretrained initialization'),
    }


class TrainRefModelArgs(TrainingArgs):
    ARGS = {
        'model_save_path': FinalModelPath.ARGS['final_model_path']
    }


# Arguments exclusively for subnetwork_selection.py:


class RetrainingArgs(TrainingArgs):
    ARGS = {
        'model_save_path':
            Argument('--retrain-model-path', type=str, default='models/retrained-model.pth',
                     help='path to save the retrained model to')
    }


class SharedPruneArgs(ArgumentClass):
    ARGS = {
        'prune_ratio':
            Argument('--prune-ratio', type=float, default=0.95, help='ratio of network units to be pruned'),
        'prune_across_modules':
            Argument('--prune-across-modules', action='store_true',
                     help='if --use-threshold is not set, whether to prune a fixed amount of units '
                          'across all network modules or prune a fixed amount of units per network module')
    }


class PruneInitArgs(SharedPruneArgs):
    ARGS = {
        'prune_by':
            Argument('--init-prune-by', type=str, choices=['weight', 'weight_gradient', 'output', 'output_gradient'],
                     default='weight', help='pruning method for pruning of model at initialization'),
        'load_prune_masks':
            Argument('--init-load-prune-masks', action='store_true', help='load init model prune masks from savefile'),
        'save_prune_masks':
            Argument('--init-save-prune-masks', action='store_true',
                     help='save prune masks generated for init model to savefile'),
        'prune_masks_filepath':
            Argument('--init-prune-masks-filepath', type=str, default='prune/init-prune-masks.npz',
                     help='path to savefile for saving/loading init model prune masks')
    }


class PruneFinalArgs(SharedPruneArgs):
    ARGS = {
        'prune_by':
            Argument('--final-prune-by', type=str, choices=['weight', 'weight_gradient', 'output', 'output_gradient'],
                     default='weight', help='pruning method for pruning of final model'),
        'load_prune_masks':
            Argument('--final-load-prune-masks', action='store_true', help='load final model prune masks from savefile'),
        'save_prune_masks':
            Argument('--final-save-prune-masks', action='store_true',
                     help='save prune masks generated for final model to savefile'),
        'prune_masks_filepath':
            Argument('--final-prune-masks-filepath', type=str, default='prune/final-prune-masks.npz',
                     help='path to savefile for saving/loading final model prune masks')
    }


class LoadModelArgs(InitModelPath, FinalModelPath, Architecture):
    ARGS = {}


class ExperimentArgs(Seed, LoadModelArgs):
    ARGS = {
        'retrain':
            Argument('--retrain', action='store_true', help='retrain the masked subnetwork of the initialized model'),
        'use_final_subnetwork':
            Argument('--use-init-subnetwork', action='store_false',
                     help='use the prune mask obtained from initialized model rather than from the final model'),
        'save_results':
            Argument('--save-results', action='store_true', help='save pruning results'),
        'results_filepath':
            Argument('--results-filepath', type=str, default='prune/prune-results.npz',
                     help='path to saved prune results')
    }


# Arguments exclusively for subnetwork_selection.sparse_grad_training


class SparseTrainingArgs(TrainingArgs):
    ARGS = {
        'model_save_path':
            Argument('--sparse-train-model-path', type=str, default='models/sparse-trained-model.pth',
                     help='path to save the sparse-gradient-training model to'),
        'mean_max_coef':
            Argument('--mean-max-coef', type=float, default=6,
                     help='weighting of mean gradient magnitude against max gradient magnitude. '
                          'Used in approximation of percentile point to use for gradient thresholding')
    }


# Arguments exclusively for distributed_processing.py


class DistributedTrainingArgs(TrainingArgs):
    ARGS = {
        'model_save_path':
            Argument('--distributed-model-path', type=str, default='models/distributed-model.pth',
                     help='path to save the distributed model to')
    }


# Arguments exclusively for gradient alignment experiment


class GradAlignmentArgs(TrainingArgs):
    ARGS = {
        'test_epochs':
            Argument('--test-epochs', type=int, nargs='+', default=[0, 1, 5, 20, 60, 120, 160],
                     help='epochs to test for gradient alignment'),
        'test_layers':
            Argument('--test-layers', type=str, nargs='+',
                     default=['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer4.0.conv1', 'layer4.1.conv2'],
                     help='layers to track gradient alignment for')
    }


all_argsets = [
    NumClass, Seed, Architecture, DataArgs, InitModelPath, FinalModelPath, TrainingArgs,
    ModelInitArgs, TrainRefModelArgs, RetrainingArgs, DistributedTrainingArgs, GradAlignmentArgs,
    SharedPruneArgs, PruneInitArgs, PruneFinalArgs,
    ExperimentArgs, SparseTrainingArgs, FeatureDataArgs
]
