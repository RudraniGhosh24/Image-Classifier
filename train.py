# train.py
import argparse
from model_utils import build_model, train_model, save_checkpoint
from utils import load_data, process_image
from workspace_utils import active_session  

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a dataset.')
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (e.g., "vgg16")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    with active_session():
        dataloaders, dataset_sizes = load_data(args.data_directory)

        model = build_model(args.arch, args.hidden_units)
        train_model(model, dataloaders, dataset_sizes, args.learning_rate, args.epochs, args.gpu, args.save_dir)
        save_checkpoint(model, args.save_dir)

if __name__ == "__main__":
    main()
