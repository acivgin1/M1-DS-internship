import csv
import numpy as np
import operator

titles = ['Mr', 'Mrs', 'Miss', 'Ms', 'Master',
          'Madame', 'Mlle',
          'Captain', 'Colonel', 'Col', 'Major',
          'Sir', 'Lady',
          'Dr',
          'Rev']
col = ['Captain', 'Colonel', 'Col', 'Major']
mrs = ['Mrs', 'Madame']
ms = ['Ms', 'Miss', 'Mlle']

older_male = ['Mr', 'Col', 'Dr', 'Rev', 'Sir']

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']


def is_spouse(husband, wife):
    return husband.surname == wife.surname and \
           husband.name.split(' ')[0] == wife.spouse_name and \
           husband.title in older_male


def repack_csv(csvreader):
    pass_list = [Passenger(row) for row in csvreader]

    # adding onboard spouse id, spouse name is later ignored since it was
    # probably irrelevant to the survival probability

    for passenger, ind in zip(pass_list, range(len(pass_list))):
        passenger.id = ind
        probable_spouse = False
        if len(passenger.name.split(' ')) > 3:  # this could be a potentional spouse
            passenger.spouse_name = passenger.name.split(' ')[0]
            probable_spouse = True
            # this is safe because we later ommit the spouse name attribute
        if passenger.title == 'Lady':
            spouse_name = [x.name for x in pass_list if x.title == 'Sir' and x.surname == passenger.surname]
            if spouse_name:
                passenger.spouse_name = spouse_name[0].split(' ')[0]

        if passenger.spouse_name:
            spouse = next((x for x in pass_list if is_spouse(x, passenger)), None)
            if spouse:
                spouse.spouse_name = passenger.name.split(' ')[0]
                spouse.spouse_id = pass_list.index(passenger)
                passenger.spouse_id = pass_list.index(spouse)
                if probable_spouse:
                    passenger.name = ' '.join(passenger.name.split(' ')[2:])

    # age estimation based on the median age of a specific passenger title and p_class set
    # firstly we remove the badly loaded data and change it with the previous neighbour value
    bad_data = [x for x in pass_list if not '1' <= x.p_class <= '3']
    for bad_elem in bad_data:
        bad_elem.p_class = pass_list[bad_elem.id - 1].p_class

    for elem in pass_list:
        elem.p_class = int(elem.p_class)

    for title in titles_final:
        for i in range(1, 4):
            age_list = np.array([x.age for x in pass_list if x.title == title and x.p_class == i and x.age > 0])

            if i == 2 or i == 3:
                if title == 'Master' or title == 'Mrs' or title == 'Ms':
                    print('class: {}, title: {}, est_age: {}'.format(i, title, np.median(age_list)))

            for elem in pass_list:
                if elem.age == -1:
                    elem.age = round(np.median(age_list) + 0.9 * np.std(age_list) * np.random.randn(1)[0])

    # title estimation based on age and sex
    median_title_age = []
    for title in titles_final[0:4]:
        age_list = np.array([x.age for x in pass_list if x.title == title])
        median_title_age.append(np.median(age_list))

    #           Mr      Mrs     Ms      Master
    # mean:     36.35,  38.5,   29.7,   20.9
    # median:   41,     41,     36,     11
    # a Master is someone who is male and younger than 3/4 * 11 + 1/4 * 41 = 21
    # a Miss is someone who is female and younger than 3/4 * 36 + 1/4 * 41 = 37

    ms_age_threshold = (median_title_age[1] + median_title_age[2] * 3) / 4
    mr_age_threshold = (median_title_age[0] + median_title_age[3] * 3) / 4

    for elem in pass_list:
        if not elem.title:
            if elem.sex == 'male':
                elem.title = 'Mr' if elem.age > mr_age_threshold else 'Master'
            else:
                elem.title = 'Mrs' if elem.age > ms_age_threshold else 'Ms'

    # find family relations,if a person is onboard with someone with the same surname and
    # is in the same passenger class (families always travel together) they're probably related
    for elem in pass_list:
        if elem.family_id:
            continue

        possible_elem_fam_list = [x for x in pass_list if x.surname == elem.surname and x.p_class == elem.p_class]
        if len(possible_elem_fam_list) > 1:
            if not elem.p_class == 3 or len(possible_elem_fam_list) < 5:
                possible_elem_fam_list.sort(key=operator.attrgetter('age'), reverse=True)
                possible_elem_fam_ids = [x.id for x in possible_elem_fam_list]

                for ids in possible_elem_fam_ids:
                    pass_list[ids].family_id = possible_elem_fam_ids[0]
                    pass_list[ids].fam_size = len(possible_elem_fam_ids)

    # save the extracted data to a new csv file
    with open('Data/Titanic_dataset_extended_1.csv', 'w', newline='') as csvfile:
        fieldnames = ['surname', 'title', 'name', 'age', 'sex', 'p_class', 'fam_size', 'survived']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for elem in pass_list:
            writer.writerow(elem.to_dictionary(fieldnames))


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
            self.title = rest.split(' ')[0] if rest.split(' ')[0] in titles else 'Master'
            if self.title == 'the':
                self.title = 'Lady'

            self.name = rest.split('(')[1][:-1]
            self.spouse_name = rest.split(' ')[1].split(' ')[0]  # we just save the first name
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
        self.family_id = None

    def __str__(self):
        return '{}, {} {} a:{} s:{} sur:{}'.format(self.title, self.surname, self.name,
                                                   self.age, self.sex, self.survived)

    def to_dictionary(self, fieldnames):
        attr_list = [self.__dict__[x] for x in fieldnames]
        return dict(zip(fieldnames, attr_list))


if __name__ == '__main__':
    with open('Data/Titanic_dataset.csv') as csvfile:
        titanic_reader = csv.DictReader(csvfile)
        repack_csv(titanic_reader)
