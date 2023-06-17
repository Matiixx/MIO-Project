from model import create_custom_model, predict_populatiry
from utils import reduce_dataset
import sys


def main(argv):
    if argv == "create_model":
        create_custom_model("data/reduced_data.csv")
    elif argv == "reduce_dataset":
        reduce_dataset("data/trumptweets.csv")
    elif argv == "predict_popularity":
        print(
            predict_populatiry(
                '"My persona will never be that of a wallflower - I’d rather build walls than cling to them" --Donald J. Trump'
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
