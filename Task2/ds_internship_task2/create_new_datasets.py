import os
import csv

import numpy as np

DATASET_FILENAME = 'Data/Titanic_dataset'

TITLES = ['Mr', 'Mrs', 'Miss', 'Ms', 'Master',
          'Madame', 'Mlle',
          'Captain', 'Colonel', 'Col', 'Major',
          'Sir', 'Lady',
          'Dr',
          'Rev']
COL = ['Captain', 'Colonel', 'Col', 'Major']
MRS = ['Mrs', 'Madame']
MS = ['Ms', 'Miss', 'Mlle']

TITLES_FINAL = ['Mr', 'Mrs', 'Ms', 'Master', 'Dr', 'Rev', 'Col', 'Sir', 'Lady']


class Passenger:
    def __init__(self, csv_row):
        self.id = None
        self.fam_size = 0
        self.surname = csv_row['Name'].split(',')[0].split('(')[0]
        # there is the example of surname (surname_alt) with surname_alt being the alternative surname pronounciation
        # or the maiden name, we ommit it

        if len(csv_row['Name'].split(',')) > 1:
            rest = csv_row['Name'].split(',')[1][1:]
        else:
            self.surname = csv_row['Name'].split(' ')[0]
            rest = ' '.join(csv_row['Name'].split(' ')[1:])

        if len(rest.split('(')) > 1:
            self.title = rest.split(' ')[0] if rest.split(' ')[0] in TITLES else 'Master'
            if self.title == 'the':
                self.title = 'Lady'

            self.name = rest.split('(')[1][:-1]
            self.spouse_name = rest.split(' ')[1].split(' ')[0]  # we just save the first name
        else:
            title_name = rest.split(' ')
            self.title = title_name[0] if title_name[0] in TITLES else ''

            self.name = ''
            if len(title_name) > 1:
                self.name = ' '.join(title_name[1:]) if self.title else ' '.join(title_name)

            self.spouse_name = ''

        self.age = csv_row['Age']
        self.age = -1 if self.age == 'NA' else float(self.age)

        self.sex = csv_row['Sex']
        self.survived = csv_row['Survived']
        self.p_class = csv_row['PClass'][0]

        self.title = 'Col' if self.title in COL else self.title
        self.title = 'Mrs' if self.title in MRS else self.title
        self.title = 'Ms' if self.title in MS else self.title

    def to_dictionary(self, fieldnames):
        attr_list = [self.__dict__[x] for x in fieldnames]
        return dict(zip(fieldnames, attr_list))


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_new_dataset(filename, pass_list, num_of_testing, mean, std_dev_scale, better_title):
    for elem, i in zip(pass_list, range(len(pass_list))):
        elem.id = i

    bad_data = [x for x in pass_list if not '1' <= x.p_class <= '3']
    for bad_elem in bad_data:
        bad_elem.p_class = pass_list[bad_elem.id - 1].p_class

    for elem in pass_list:
        elem.p_class = int(elem.p_class)

    for title in TITLES_FINAL:
        for i in range(1, 4):
            age_list = np.array([x.age for x in pass_list if x.title == title and x.p_class == i and x.age > 0])
            for elem in pass_list:
                if elem.age == -1 and elem.title == title and elem.p_class == i:
                    base = np.mean(age_list) if mean else np.median(age_list)
                    elem.age = base + std_dev_scale * np.std(age_list) * np.random.randn(1)[0]

    median_title_age = []
    for title in TITLES_FINAL[0:4]:
        age_list = np.array([x.age for x in pass_list if x.title == title])
        median_title_age.append(np.median(age_list))

    ms_age_threshold = (median_title_age[1] + median_title_age[2] * 3) / 4
    mr_age_threshold = (median_title_age[0] + median_title_age[3] * 3) / 4

    for elem in pass_list:
        if not elem.title:
            if elem.sex == 'male':
                elem.title = 'Mr' if elem.age > mr_age_threshold and better_title else 'Master'
            else:
                elem.title = 'Mrs' if elem.age > ms_age_threshold and better_title else 'Ms'

    for elem in pass_list:
        if elem.family_id:
            continue

        possible_elem_fam_ids = [x.id
                                 for x in pass_list
                                 if x.surname == elem.surname and x.p_class == elem.p_class]

        # if len(possible_elem_fam_ids) > 1:
        for ids in possible_elem_fam_ids:
            pass_list[ids].fam_size = len(possible_elem_fam_ids)

    # save the extracted data to a new csv file
    with open('{}_training.csv'.format(filename), 'w', newline='') as training, \
            open('{}_testing.csv'.format(filename), 'w', newline='') as testing:

        fieldnames = ['surname', 'title', 'name', 'age', 'sex', 'p_class', 'fam_size', 'survived']
        training_writer = csv.DictWriter(training, fieldnames=fieldnames)
        testing_writer = csv.DictWriter(testing, fieldnames=fieldnames)

        training_writer.writeheader()
        for elem in pass_list[:-num_of_testing]:
            training_writer.writerow(elem.to_dictionary(fieldnames))

        testing_writer.writeheader()
        for elem in pass_list[-num_of_testing:]:
            testing_writer.writerow(elem.to_dictionary(fieldnames))


def create_new_datasets(filename=DATASET_FILENAME):
    create_dir('Data/Datasets')
    i = 1
    for mean in [False, True]:
        for std_dev_scale in [0, 0.2, 0.4, 0.6, 0.8]:
            for better_title in [False, True]:
                with open('{}.csv'.format(filename)) as training, \
                        open('{}_for_testing.csv'.format(filename)) as testing:

                    titanic_training_reader = csv.DictReader(training)
                    titanic_testing_reader = csv.DictReader(testing)

                    training_pass = [Passenger(x) for x in titanic_training_reader]
                    testing_pass = [Passenger(x) for x in titanic_testing_reader]
                    training_pass.extend(testing_pass)

                    save_to_filename = 'Data/Datasets/Titanic_{}'.format(i)
                    i += 1

                    create_new_dataset(save_to_filename,
                                       training_pass,
                                       len(testing_pass),
                                       mean,
                                       std_dev_scale,
                                       better_title)


if __name__ == '__main__':
    create_new_datasets()
