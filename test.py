import STACCI
import argparse


parser = argparse.ArgumentParser(description="STACCI for CCI prediction on Spatial-Omics") # Tid: HACK
parser.add_argument('--root', required=True, help="which root to do CCI prediction", type=str)
parser.add_argument('--ds-dir', required=True, type=str, metavar='DATA', help='dataset directory')
parser.add_argument('--ds-name', required=True, type=str, metavar='DATA', help='dataset name')
parser.add_argument('--h5-name', required=True, type=str, metavar='DATA', help='h5ad file name')
test_args = parser.parse_args()


if __name__ == '__main__':
    # Test
    args = STACCI.prepare(test_args.root, test_args.ds_dir, test_args.ds_name, test_args.h5_name)
    # args = prepare('../tests/', 'datasets/', 'T25_F1', 'T25_F1_1000hvg_ceco')
    STACCI.train(args)