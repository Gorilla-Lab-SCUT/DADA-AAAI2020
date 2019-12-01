import argparse


def opts():
    parser = argparse.ArgumentParser(description='DADA-P', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data options
    parser.add_argument('--source-train-data-path', type=str, help='Root of train data set of the source domain')
    parser.add_argument('--target-train-data-path', type=str, help='Root of train data set of the target domain')
    parser.add_argument('--target-test-data-path', type=str, help='Root of test data set of the target domain')
    parser.add_argument('--source-domain', type=str, help='Source domain')
    parser.add_argument('--target-domain', type=str, help='Target domain')
    parser.add_argument('--num-classes-s', type=int, help='Number of classes of the source domain')
    # Optimization options
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='The initial learning rate')
    parser.add_argument('--train-by-iter', action='store_true', help='Whether to change the learning rate and lambda by iteration or epoch')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay (L2 penalty)')
    parser.add_argument('--lam', action='store_true', help='Whether to use lambda or not (lambda=1)')
    parser.add_argument('--convex-combine', action='store_true', help='Whether to use a convex combination of the normalized vector and an all-ones vector 1 as the class weight vector')
    parser.add_argument('--disc-tar', action='store_true', help='Whether to use discriminative adversarial learning on the target data')
    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restart)')
    parser.add_argument('--stop-epoch', default=200, type=int, metavar='N', help='Stop epoch (default: 200)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoint path to resume')
    parser.add_argument('--pretrained-checkpoint', type=str, default='', help='Source pre-trained checkpoint to resume')
    parser.add_argument('--test-only', action='store_true', help='Test only flag')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet', help='Model name')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Whether to use ImageNet pre-trained model')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('--no-da', action='store_true', help='Whether to use data augmentation')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='Number of data loading workers (default: 4)')
    parser.add_argument('--test-time', default=200, type=int, help='Test times (default: 200)')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='Print frequency (default: 10)')
    args = parser.parse_args()
    
    args.log = args.log + '_adapt_' + args.source_domain + '_to_' + args.target_domain + '_' + str(args.epochs) + 'epoch_bs' + str(args.batch_size) + '_' + args.arch

    return args
