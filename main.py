__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.5'
__status__ = "Research"
__date__ = "30/1/2018"
__license__= "MIT License"

import argparse
from amnet_model import *
import amnet_model as amnet_model
from amnet import *
import amnet as amnet
from config import *
from utils import *


def main():
    parser = argparse.ArgumentParser(description='AMNet Image memorability prediction with attention')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID. If -1 the application will run on CPU')
    parser.add_argument('--model-weights', default='', type=str, help='pkl file with the model weights')

    parser.add_argument('--cnn', default='ResNet50FC', type=str, help='Name of CNN model for features extraction [ResNet18FC, ResNet50FC, ResNet101FC, VGG16FC, ResNet50FT]')
    parser.add_argument('--att-off', action="store_true", help='Runs training/testing without the visual attention')
    parser.add_argument('--lstm-steps', default=3, type=int,
                        help='Number of LSTM steps. Default 3. To disable LSTM set to zero')

    parser.add_argument('--last-step-prediction', action="store_true",
                        help='Predicts memorability only at the last LSTM step')

    parser.add_argument('--test', action='store_true', help='Run evaluation')

    parser.add_argument('--eval-images', default='', type=str, help='Directory or a csv file with images to predict memorability for')
    parser.add_argument('--csv-out', default='', type=str, help='File where to save prediced memorabilities in csv format')
    parser.add_argument('--att-maps-out', default='', type=str, help='Directory where to store attention maps')

    # Training
    parser.add_argument('--epoch-max', default=-1, type=int,
                        help='If not specified, number of epochs will be set according to selected dataset')
    parser.add_argument('--epoch-start', default=0, type=int,
                        help='Allows to resume training from a specific epoch')

    parser.add_argument('--train-batch-size', default=-1, type=int,
                        help='If not specified a default size will be set according to selected dataset')
    parser.add_argument('--test-batch-size', default=-1, type=int,
                        help='If not specified a default size will be set according to selected dataset')

    # Dataset configuration
    parser.add_argument('--dataset', default='lamem', type=str, help='Dataset name [lamem, sun]')
    parser.add_argument('--experiment', default='', type=str, help='Experiment name. Usually no need to set' )
    parser.add_argument('--dataset-root', default='', type=str, help='Dataset root directory')
    parser.add_argument('--images-dir', default='images', type=str, help='Relative path to the test/train images')
    parser.add_argument('--splits-dir', default='splits', type=str, help='Relative path to directory with split files')
    parser.add_argument('--train-split', default='', type=str, help='Train split filename e.g. train_2')
    parser.add_argument('--val-split', default='', type=str, help='Validation split filename e.g. val_2')
    parser.add_argument('--test-split', default='', type=str, help='Test split filename e.g. test_2')

    args = parser.parse_args()
    hps = get_amnet_config(args)

    print("Configuration")
    print("----------------------------------------------------------------------")
    print(hps)

    amnet = AMNet()
    amnet.init(hps)

    if hps.test_split != '':
        split_files = get_split_files(hps.dataset_root, hps.splits_dir, hps.test_split)
        if hps.model_weights == '':
            weight_files = get_weight_files(split_files, experiment_name=hps.experiment_name, max_rc_checkpoints=True)
        else:
            weight_files = [hps.model_weights]

        print("Splits: ", split_files)
        print("Model weights: ", weight_files)
        amnet.eval_models(weight_files, split_files)
        return


    if hps.eval_images != '':
        if hps.model_weights == '' or not os.path.isfile(hps.model_weights):
            print("You need to specify path to model weights with parameter --model-weights")
            return

        print("Images filename/path: ", hps.eval_images)
        print("Model weights: ", hps.model_weights)
        result = amnet.predict_memorability(hps.eval_images)

        result.write_stdout()
        if args.csv_out != '':
            print("Saving memorabilities to:",args.csv_out)
            result.write_csv(args.csv_out)

        if args.att_maps_out != '':
            print("Saving attention maps to:", args.att_maps_out)
            result.write_attention_maps(args.att_maps_out)

        return

    # Training phase
    amnet.train()


if __name__ == "__main__":
    print_pkg_versions()
    main()