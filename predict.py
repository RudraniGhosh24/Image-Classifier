# predict.py
import argparse
from model_utils import load_checkpoint, predict
from utils import process_image
from workspace_utils import active_session

def main():
    parser = argparse.ArgumentParser(description='Predict the class for an input image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    with active_session():
        model = load_checkpoint(args.checkpoint)
        input_image = process_image(args.image_path)
        probs, classes = predict(model, input_image, args.top_k, args.gpu)

        print("Probabilities:", probs)
        print("Classes:", classes)

if __name__ == "__main__":
    main()
