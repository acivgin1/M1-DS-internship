import sys

from .create_new_datasets import create_new_datasets
from .visualise_dataset import create_visualisations
from .titanic_classification import classifier_main


def main(argv=None):
    if argv is None:
        argv = sys.argv

    data_path = argv[1]

    cnd = False
    cnv = False
    tnm = False
    prn = False
    slp = False

    for arg in argv:
        if arg == '-cdata':
            cnd = True
        if arg == '-cvis':
            cnv = True
        if arg == '-tmodels':
            tnm = True
        if arg == '-print':
            prn = True
        if arg == '-splt':
            slp = True

    if cnd:
        create_new_datasets(data_path=data_path)

    if cnv:
        create_visualisations(data_path=data_path, show_last_plot=slp)

    classifier_main(tnm, prn, data_path=data_path)


if __name__ == "__main__":
    sys.exit(main())
