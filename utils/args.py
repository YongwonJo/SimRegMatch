import argparse


def SimRegMatch_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', type=str, default='SimRegMatch')

    # For data
    parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb'], help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--labeled-ratio', type=float, default=0.1)
    parser.add_argument('--img_size', type=int, default=224, help='image size used in training')

    # For model architecture
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='resnet50', help='model name')
    
    # For uncertainty estimation
    parser.add_argument('--threshold', type=float, default=10)
    parser.add_argument('--percentile', default=0.95)    
    parser.add_argument('--iter-u', type=int, default=5)
    
    # For pseudo-label calibration
    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.6)

    # For loss calculation
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
    parser.add_argument('--lambda-u', type=float, default=0.01)
    
    # For model training
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')    
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='optimizer weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    # seed
    parser.add_argument('--seed', default=0)
    
    return parser
