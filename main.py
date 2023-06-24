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
                "If Syria was forced to use Obamacare they would self-destruct without a shot being fired. Obama should sell them that idea!"
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])

# If Syria was forced to use Obamacare they would self-destruct without a shot being fired. Obama should sell them that idea! - 1045 retweets
