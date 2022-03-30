import csv
import os
import argparse
from tqdm import tqdm
from big_sleep import Imagine
from time import time

def parse_arguments():
    """Returns a command line parser

    Returns
    ----------
    argparse.Namespace

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--out_path",
                        dest="out_path",
                        help="""Path to out images""",
                        default="./gen_images",
                        type=str)

    parser.add_argument("-l", "--log_file",
                        dest="log_file",
                        help="""Logging file""",
                        default="imagine_log.lg",
                        type=str)

    parser.add_argument("-f", "--csv_file",
                        dest="csv_file",
                        help="""Path to input csv file""",
                        default="training_data/EN-English/en_train_translatednl_all.csv",
                        type=str)

   
    parser.add_argument("-t", "--tmp_file",
                        dest="tmp_file",
                        help="""Path to tmp file""",
                        default="currents.tmp",
                        type=str)
    return parser.parse_args()

def read_csv(csv_file):

    phrases = list()

    with open(csv_file) as ap:
        csvreader = csv.DictReader(ap, delimiter="\t")
        for row in csvreader:
            phrases.append([row['id'].split(" ")[2], row["phrase"]])

    return phrases


def check_exists(out_path, out_image):

    f = os.path.join(out_path, f"{out_image}.png")
    return f, os.path.isfile(f)

def write_to_currents(image_location, currents_file):
    
    with open(currents_file, "a") as ap:
        ap.write(f"{image_location}\n")


def in_currents(image_location, currents_file):
    
    with open(currents_file, "r") as ap:
        currents = [_.strip() for _ in ap.readlines()]
    
    if image_location in currents:
        return True
    else:
        return False
    
def main():
    args = parse_arguments()

    phrases = read_csv(args.csv_file)

    dream = Imagine(
        text="",
        lr=7e-2,
        save_every=50,
        iterations=500,
        epochs=10,
        save_progress=False,
        save_best=True,
        image_size=512,
        max_classes=15,
        out_path=args.out_path,
        out_name="",
        worst_limit=5
    )

    with tqdm(total=len(phrases), desc='Phrases status', leave=True, initial=1) as pbar:
        for id, phrase in phrases:
            image_location , exists = check_exists(args.out_path, id)
            if exists or in_currents(image_location, args.tmp_file):
                print(f"{image_location} already imagined.")
            else:
                write_to_currents(image_location, args.tmp_file)       
                s_time = time()
                dream.reset()
                dream.set_text(text=phrase, out_name=id)
                dream()
                e_time = time()
                with open(args.log_file, "a") as ap:
                    ap.write(f"{image_location}\t{(e_time-s_time)/60}[m]\n")
                os.rename(os.path.join(args.out_path, f"{id}.best.png"), os.path.join(args.out_path, f"{id}.final.png"))
            pbar.update(1)

if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()
