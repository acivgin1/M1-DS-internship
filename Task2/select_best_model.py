with open('gridSearchCV.txt', 'r') as grid_searched:
    i = 0
    continue_print = 0
    max_score = 0.0
    max_id = 0
    for line in grid_searched:
        if line[0:4] == 'mean':
            print('')
            print(line)
        if line[0:5] == 'scale':
            print('')
            print(line)
        if line[0:13] == 'Logistic Regr' or continue_print:
            if not continue_print:
                continue_print = 0
            print(line, end='')
            if continue_print == 3:
                score = float(line.split(':')[1][1:-1])
                if score > max_score:
                    max_score = score
                    max_id = i

            continue_print += 1

            if continue_print > 3:
                continue_print = 0
            continue
        i += 1

print('Max score: {}, max_id: {}'.format(max_score, max_id))

# Decision tree
# mean[True], std_dev_scale[0.6], better_title[False]
# scale[on], fam_size[on], group_age[on]
#
# {'max_depth': 9}
# Best score is: 0.6971409473950594
#
# Support Vector Machine
# mean[True], std_dev_scale[0.2], better_title[True]
# scale[off], fam_size[on], group_age[on].
#
# {'C': 3, 'gamma': 0.01}
# Best score is: 0.7053678518544504
#
# Gaussian Naive Bayes
# mean[False], std_dev_scale[0.6], better_title[True]
# scale[on], fam_size[on], group_age[off].
# Best score is: 0.7390636991557943
#
# scale[off], fam_size[on], group_age[off].
# Best score is: 0.7390636991557943
#
# Logistic Regression
# mean[True], std_dev_scale[0.2], better_title[False]
# scale[off], fam_size[on], group_age[off].
# {'C': 10}
# Best score is: 0.6148514163742468
