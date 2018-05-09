import csv
import numpy as np

titles = ['Mr', 'Mrs', 'Miss', 'Ms', 'Master',
          'Madame', 'Mlle',
          'Captain', 'Colonel', 'Col', 'Major',
          'Sir', 'Lady',
          'Dr',
          'Rev']
col = ['Captain', 'Colonel', 'Col', 'Major']
mrs = ['Mrs', 'Madame']
ms = ['Ms', 'Miss', 'Mlle']

older_male = ['Mr', 'Col', 'Dr', 'Rev']

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']


def is_spouse(husband, wife):
    return husband.surname == wife.surname and \
           husband.name.split(' ')[0] == wife.spouse_name and \
           husband.title in older_male


def repack_csv(csvreader):
    pass_list = [Passenger(row) for row in csvreader]

    # adding spouse id onboard
    for passenger, ind in zip(pass_list, range(len(pass_list))):
        passenger.id = ind
        if passenger.spouse_name:
            spouse = next((x for x in pass_list if is_spouse(x, passenger)), None)
            if spouse:
                spouse.spouse_name = passenger.name.split(' ')[0]
                spouse.spouse_id = pass_list.index(passenger)
                passenger.spouse_id = pass_list.index(spouse)

    # estimate age based on title and passenger class
    bad_data = [x for x in pass_list if not '1' <= x.p_class <= '3']
    for bad_elem in bad_data:
        ind = pass_list.index(bad_elem)
        bad_elem.p_class = pass_list[ind-1].p_class

    for elem in pass_list:
        elem.p_class = int(elem.p_class)

    for title in titles_final:
        for i in range(1, 3):
            age_list = np.array([x.age for x in pass_list if x.title == title and x.p_class == i and x.age > 0])
            for elem in pass_list:
                if elem.age == -1:
                    elem.age = np.median(age_list)
    with open('Task2/Data/Titanic_dataset_extended.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'surname', 'title', 'name', 'age', 'sex', 'p_class', 'spouse_id', 'survived']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for elem in pass_list:
            writer.writerow(elem.to_dictionary(fieldnames))


class Passenger:
    def __init__(self, csv_row):
        self.id = None
        self.surname = csv_row['Name'].split(',')[0].split('(')[0]
        # there is the example of surname (surname_alt) with surname_alt being the alternative surname pronounciation
        # or the maiden name, we ommit it

        if len(csv_row['Name'].split(',')) > 1:
            rest = csv_row['Name'].split(',')[1][1:]
        else:
            self.surname = csv_row['Name'].split(' ')[0]
            rest = ' '.join(csv_row['Name'].split(' ')[1:])

        if len(rest.split('(')) > 1:
            self.title = rest.split(' ')[0]
            if self.title == 'the':
                self.title = 'Lady'

            self.name = rest.split('(')[1][:-1]
            self.spouse_name = rest.split(' ')[1].split(' ')[0]     # we just save the first name
        else:
            title_name = rest.split(' ')
            self.title = title_name[0] if title_name[0] in titles else ''

            self.name = ''
            if len(title_name) > 1:
                self.name = ' '.join(title_name[1:]) if self.title else ' '.join(title_name)

            self.spouse_name = ''

        self.age = csv_row['Age']
        self.age = -1 if self.age == 'NA' else float(self.age)

        self.sex = csv_row['Sex']
        self.survived = csv_row['Survived']
        self.p_class = csv_row['PClass'][0]

        self.title = 'Col' if self.title in col else self.title
        self.title = 'Mrs' if self.title in mrs else self.title
        self.title = 'Ms' if self.title in ms else self.title

        self.spouse_id = None

    def __str__(self):
        return '{}, {} {} a:{} s:{} sur:{}'.format(self.title, self.surname, self.name,
                                                   self.age, self.sex, self.survived)

    def to_dictionary(self, fieldnames):
        attr_list = [self.__dict__[x] for x in fieldnames]
        return dict(zip(fieldnames, attr_list))


if __name__ == '__main__':
    with open('Task2/Data/Titanic_dataset.csv') as csvfile:
        titanic_reader = csv.DictReader(csvfile)
        repack_csv(titanic_reader)

