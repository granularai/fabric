import argparse

def get_parser_with_args():
    parser = argparse.ArgumentParser(description='Training change detection network')
    parser.add_argument('--patch_size', type=int, default=120, required=False, help='input patch size')
    parser.add_argument('--stride', type=int, default=10, required=False, help='stride at which to sample patches')
    parser.add_argument('--aug', default=True, required=False, help='Do augmentation or not')

    parser.add_argument('--gpu_ids', default='0,1,2,3', required=False, help='gpus ids for parallel training')
    parser.add_argument('--num_workers', type=int, default=90, required=False, help='Number of cpu workers')

    parser.add_argument('--epochs', type=int, default=10, required=False, help='number of eochs to train')
    parser.add_argument('--batch_size', type=int, default=256, required=False, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate')

    parser.add_argument('--loss', type=str, default='bce', required=False, help='bce,focal,dice,jaccard,tversky')
    parser.add_argument('--gamma', type=float, default=2, required=False, help='if focal loss is used pass gamma')
    parser.add_argument('--alpha', type=float, default=0.5, required=False, help='if tversky loss is used pass alpha')
    parser.add_argument('--beta', type=float, default=0.5, required=False, help='if tversky loss is used pass beta')

    parser.add_argument('--val_cities', default='0,1', required=False, help='''cities to use for validation,
                                0:abudhabi, 1:aguasclaras, 2:beihai, 3:beirut, 4:bercy, 5:bordeaux, 6:cupertino, 7:hongkong, 8:mumbai,
                                9:nantes, 10:paris, 11:pisa, 12:rennes, 14:saclay_e''')

    parser.add_argument('--dataset', type=str, required=True, help='gcs dataset file or directory')
    parser.add_argument('--data_dir', default='../datasets/onera/', required=False, help='data directory for training')
    parser.add_argument('--weight_dir', default='../weights/', required=False, help='directory to save weights')
    parser.add_argument('--log_dir', default='../logs/', required=False, help='directory to save training log')
    parser.add_argument('--validation_city', default='rennes', required=False, help='city to output complete results')

    return parser
