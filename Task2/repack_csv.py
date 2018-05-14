import csv
import numpy as np
import operator
import matplotlib.pyplot as plt

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

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Dr', 'Rev', 'Col', 'Sir', 'Lady']


def is_spouse(husband, wife):
    return husband.surname == wife.surname and \
           husband.name.split(' ')[0] == wife.spouse_name and \
           husband.title in older_male


def plot_age_histograms(pass_list, name):
    for i in range(1, 4):
        ax = plt.figure()
        legends = []
        for title in titles_final[0:4]:
            age_list = np.array([x.age for x in pass_list if x.title == title and x.p_class == i and x.age > 0])
            legends.append('{} Class: {} Title: {}'.format(name, i, title))
            plt.hist(age_list, alpha=0.5, bins=15)
        plt.legend(legends)
        plt.title('{}. Class PDF'.format(i))
        plt.xlabel('Age [years]')
        plt.ylabel('Age frequency')
        ax.savefig('Data/Images/{}{}_pclass histogram.png'.format(i, name), bbox_inches='tight')


def repack_csv(filename, pass_list, num_of_testing, mean=True, std_dev_scale=0.2, better_title=True):
    # # adding onboard spouse id, spouse name is later ignored since it was
    # # probably irrelevant to the survival probability
    # for passenger in pass_list:
    #     probable_spouse = False
    #     if len(passenger.name.split(' ')) > 3:  # this could be a potentional spouse
    #         passenger.spouse_name = passenger.name.split(' ')[0]
    #         probable_spouse = True
    #         # this is safe because we later ommit the spouse name attribute
    #     if passenger.title == 'Lady':
    #         spouse_name = [x.name for x in pass_list if x.title == 'Sir' and x.surname == passenger.surname]
    #         if spouse_name:
    #             passenger.spouse_name = spouse_name[0].split(' ')[0]
    #
    #     if passenger.spouse_name:
    #         spouse = next((x for x in pass_list if is_spouse(x, passenger)), None)
    #         if spouse:
    #             spouse.spouse_name = passenger.name.split(' ')[0]
    #             spouse.spouse_id = pass_list.index(passenger)
    #             passenger.spouse_id = pass_list.index(spouse)
    #             if probable_spouse:
    #                 passenger.name = ' '.join(passenger.name.split(' ')[2:])

    for elem, i in zip(pass_list, range(len(pass_list))):
        elem.id = i

    # age estimation based on the median age of a specific passenger title and p_class set
    # firstly we remove the badly loaded data and change it with the previous neighbour value
    bad_data = [x for x in pass_list if not '1' <= x.p_class <= '3']
    for bad_elem in bad_data:
        bad_elem.p_class = pass_list[bad_elem.id - 1].p_class

    for elem in pass_list:
        elem.p_class = int(elem.p_class)

    # plot_age_histograms(pass_list, 'before')
    for title in titles_final:
        for i in range(1, 4):
            age_list = np.array([x.age for x in pass_list if x.title == title and x.p_class == i and x.age > 0])
            for elem in pass_list:
                if elem.age == -1 and elem.title == title and elem.p_class == i:
                    base = np.mean(age_list) if mean else np.median(age_list)
                    elem.age = base + std_dev_scale * np.std(age_list) * np.random.randn(1)[0]

    # plot_age_histograms(pass_list, 'after')

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
                elem.title = 'Mr' if elem.age > mr_age_threshold and better_title else 'Master'
            else:
                elem.title = 'Mrs' if elem.age > ms_age_threshold and better_title else 'Ms'

    # find family relations,if a person is onboard with someone with the same surname and
    # is in the same passenger class (families always travel together) they're probably related
    for elem in pass_list:
        if elem.family_id:
            continue

        possible_elem_fam_list = [x for x in pass_list if x.surname == elem.surname and x.p_class == elem.p_class]
        if len(possible_elem_fam_list) > 1:
            possible_elem_fam_list.sort(key=operator.attrgetter('age'), reverse=True)
            possible_elem_fam_ids = [x.id for x in possible_elem_fam_list]

            for ids in possible_elem_fam_ids:
                # pass_list[ids].family_id = possible_elem_fam_ids[0]
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


def main():
    i = 1
    for mean in [False, True]:
        for std_dev_scale in [0, 0.2, 0.4, 0.6, 0.8]:
            for better_title in [False, True]:
                with open('Data/Titanic_dataset.csv') as training, open('Data/Titanic_dataset_for_testing.csv') as testing:
                    titanic_training_reader = csv.DictReader(training)
                    titanic_testing_reader = csv.DictReader(testing)

                    training_pass = [Passenger(x) for x in titanic_training_reader]
                    testing_pass = [Passenger(x) for x in titanic_testing_reader]
                    training_pass.extend(testing_pass)

                    filename = 'Data/Datasets/Titanic_{}'.format(i)
                    i += 1

                    repack_csv(filename, training_pass, len(testing_pass), mean, std_dev_scale, better_title)


if __name__ == '__main__':
    main()
