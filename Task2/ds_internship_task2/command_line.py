import sys

from .create_new_datasets import create_new_datasets
from .visualise_dataset import create_visualisations
from .classification import main as class_main


def main(argv=None):
    if argv is None:
        argv = sys.argv

    cnd = False
    cnv = False
    tnm = False
    prn = False

    for arg in sys.argv:
        if arg == 'create-datasets':
            cnd = True
        if arg == 'create-visualisations':
            cnv = True
        if arg == 'train-models':
            tnm = True
        if arg == 'print':
            prn = True

    if cnd:
        create_new_datasets()

    if cnv:
        create_visualisations()

    class_main(tnm, prn)


if __name__ == "__main__":
    sys.exit(main())