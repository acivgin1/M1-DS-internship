import pandas as pd

titles = ['Mr', 'Mrs', 'Miss', 'Ms', 'Master',
          'Madame', 'Mlle',
          'Captain', 'Colonel', 'Col', 'Major',
          'Sir', 'Lady',
          'Dr',
          'Rev']

older_male = ['Mr', 'Col', 'Dr', 'Rev', 'Sir']


def extract_name_data(elem):
    col = ['Captain', 'Colonel', 'Col', 'Major']
    mrs = ['Mrs', 'Madame']
    ms = ['Ms', 'Miss', 'Mlle']

    elem['Surname'] = elem['Name'].split(',')[0].split('(')[0]
    # there is the example of surname (surname_alt), name1 with surname_alt being the alternative surname pronounciation
    # or the maiden name, we ommit it by choosing only the first name
    if len(elem['Name'].split(',')) > 1:
        rest = elem['Name'].split(',')[1][1:]
    else:
        elem['Surname'] = elem['Name'].split(' ')[0]
        rest = ' '.join(elem['Name'].split(' ')[1:])

    if len(rest.split('(')) > 1:
        elem['Title'] = rest.split(' ')[0] if rest.split(' ')[0] in titles else ''

        elem['Name'] = rest.split('(')[1][:-1]
        elem['Spouse_name'] = rest.split(' ')[1].split(' ')[0]
    else:
        title_name = rest.split(' ')
        elem['Title'] = title_name[0] if title_name[0] in titles else ''

        elem['Name'] = ''
        if len(title_name) > 1:
            elem['Name'] = ' '.join(title_name[1:] if elem['Title'] else ' '.join(title_name))

        elem['Spouse_name'] = ''

    elem['Title'] = 'Col' if elem['Title'] in col else elem['Title']
    elem['Title'] = 'Mrs' if elem['Title'] in mrs else elem['Title']
    elem['Title'] = 'Ms' if elem['Title'] in ms else elem['Title']
    return elem


def add_name_data(data):
    return data.apply(extract_name_data, axis=1)


def is_spouse(husband, wife):
    return husband.Surname == wife.Surname and \
           husband.Name.split(' ')[0] == wife.Spouse_name and \
           husband.Title in older_male


def find_spouses(elem, data):
    probable_spouse = False
    if len(elem['Name'].split(' ')) > 3:
        elem['Spouse_name'] = elem['Name'].split(' ')[0]
        probable_spouse = True

    if elem['Title'] == 'Lady':
        spouse_name = data.loc[(data['Title'] == 'Sir') & (data['Surname'] == elem['Surname'])]['Name']
        elem['Spouse_name'] = spouse_name.values[0].split(' ')[0]

    if elem['Spouse_name']:
        spouse_id = data.apply(lambda x: is_spouse(x, elem), axis=1).idxmax()
        if spouse_id:
            elem['Spouse_id'] = spouse_id
            if probable_spouse:
                elem['Name'] = ' '.join(elem['Name'].split(' ')[2:])
    return elem


df = pd.read_csv('Data/Titanic_dataset.csv', na_values='')
df = add_name_data(df)
df = df.apply(lambda x: find_spouses(x, df), axis=1)
