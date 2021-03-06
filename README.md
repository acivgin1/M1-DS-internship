
# ABH DS internship
During the three months of the internship the goal was to solve
different tasks ranging from classical data science methods, to newer
ones, including sparse SVD, etc. To deal with larger, as well as smaller
datasets, handle preprocessing of data and cross-validating, training
and testing different models.

## Table of contents
1. [Month 1 Task 2](#month-1-task-2)
	1. [Solution](#solution)
	    1. [Create new datasets](#create-new-datasets)
	    2. [Visualise dataset](#visualise-dataset)
	    3. [Titanic classification](#titanic-classification)
	2. [Results](#results)
		 1. [Model parameters](#model-parameters)
		 2. [Model results](#model-results)
	3. [Prerequisites](#prerequisites)
	4. [Installing](#installing)
2. [Month 1 Task 3](#month-1-task-3)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)
## Month 1 Task 2
The task is to perform binary classification on the Titanic dataset,
given passenger data that includes:
* Passenger's surname, title, name (full name of spouse) - `Name`,
* Passenger's sex - `Sex`,
* Passenger's class - `PClass`,
* Passenger's age - `Age`.

And using this data to predict whether or not the passenger has
surivived the sinking
of the Titanic.

### Solution
#### Create new datasets
Original dataset was parsed, bad and missing data was fixed and estimated.

We extracted passenger titles from their full names, missing titles were
estimated based on their sex, age and passenger class to one of the four
categories `'Mr', 'Mrs', 'Master', 'Ms'`.
Different names for the same title were categorized, e.g. `'Ms' = ('Ms',
'Miss', 'Mlle')`

Age was estimated based on passenger class and title, different methods
of age estimation were done, using mean and median age for different
categories, as well as adding a small deviation from the mean/median age
to preserve the age-class histograms.

We also added a new feature `family_size` which represents the number
of passengers with the same surname and the same passenger class onboard
the Titanic. With the idea being that families that travel always travel
together.
(This may not always hold, especially for the 3rd passenger
class, but the models have performed better with this naive
implementation)

#### Visualise dataset
This module performs basic dataset visualisations, listed below:
* `plot_age_histograms`: create `age` histograms for different `p_class`
and `title` categories.
* `plot_age_title`: create 2D plots of those who have survived vs those
less fortunate with x axis being their age and y axis passenger title.
* `plot_age_title_class`: create a 3D scatter graph of survivors,
with an additional z axis of `p_class` and the size of the point being
determined by their familiy size.
* `group_survivors_by_params`: this creates multiindex pandas dataframes
that group survivors by different list of attributes passed to the
function
* `create_survivor_groups`: a parent function that uses the
`group_survivors_by_params` function to create groupings by different
combinations of parameters.
* `create_visualisations`: a single function that performs the following:
    * create age histograms,
    * create survivor groups dataframes,
    * 2D plot of survivors by age and title,
    * 3D plot of survivors by age, title, class and family_size.

Files generated by this function will appear in the `/Data/Images`,
`/Data/Pickles`, `/Data/Dataframes` folders.

After using this module, we get the following results
(red represents survivors):

![alt text][first_class_hist]

![alt text][second_class_hist]

![alt text][third_class_hist]

![alt text][age_title]

![alt text][age_title_class]

[first_class_hist]: https://raw.githubusercontent.com/acivgin1/M1-DS-internship/working/Task2/Data/Images/1_pclass_histogram_14.png "First class age histogram"
[second_class_hist]: https://raw.githubusercontent.com/acivgin1/M1-DS-internship/working/Task2/Data/Images/2_pclass_histogram_14.png "Second class age histogram"
[third_class_hist]: https://raw.githubusercontent.com/acivgin1/M1-DS-internship/working/Task2/Data/Images/3_pclass_histogram_14.png "Third class age histogram"
[age_title]: https://raw.githubusercontent.com/acivgin1/M1-DS-internship/working/Task2/Data/Images/Age_Title.png "Age title plot"
[age_title_class]: https://raw.githubusercontent.com/acivgin1/M1-DS-internship/working/Task2/Data/Images/Age_Title_Class_Fam_size.png "Age title class plot"

#### Titanic classification

This module performs cross validation on four different classifiers,
finding the following best hyper parameters:
* Dataset parameters:
    * `mean`: bool, represents whether we estimate age based on
    median or the mean age,
    * `std_dev_scale`: float, represents the value of the standard
    deviation scaler used for age estimation. range: (0, 1)
    * `better_title`: bool, represents whether we estimate age naively,
    based only on passenger sex or taking into consideration passenger
    title as well.
* Data format parameters:
    * `scale_data`: bool, do we need to scale the data or not,
    * `delete_fam_size`: do we take into consideration the induced family
    size parameter
    * `group_by_age`: do we convert the continous `age` parameter to
    a categorical variable.
* Model parameters, these vary for different models.

The classifiers that we use are listed below:
* Decision tree,
* Gaussian Naive Bayes,
* Support Vector Machine,
* Logistic Regression.

And finally we use a Voting Classifier, that ensembles the previously
trained classifiers, using their achieved validation accuracies as
weights.
### Results

#### Model parameters
<table>
  <tr>
    <th rowspan="2">Model name</th>
    <th rowspan="2">Model params</th>
    <th colspan="3">Data format params</th>
    <th colspan="3">Dataset params</th>
  </tr>
  <tr>
    <td>Scale data</td>
    <td>Delete fam_size</td>
    <td>Group by age</td>
    <td>Mean</td>
    <td>std_dev scaler</td>
    <td>Better title est.</td>
  </tr>
  <tr>
    <td>Decision tree</td>
    <td>max_depth: 4</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>True</td>
    <td>0.6</td>
    <td>False</td>
  </tr>
  <tr>
    <td>Gaussian Naive Bayes</td>
    <td>None</td>
    <td>False</td>
    <td>False</td>
    <td>True</td>
    <td>True</td>
    <td>0.8</td>
    <td>False</td>
  </tr>
  <tr>
    <td>Support Vector Machine</td>
    <td>C: 1, gamma:0.03</td>
    <td>True</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>0.2</td>
    <td>True</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>C: 1</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>0.4</td>
    <td>False</td>
  </tr>
  <tr>
    <td>Voting Classifier</td>
    <td>None</td>
    <td>True</td>
    <td>False</td>
    <td>True</td>
    <td>True</td>
    <td>0.2</td>
    <td>False</td>
  </tr>
</table>

#### Model results
<table>
  <tr>
    <th colspan="3">Decision Tree</th>
  </tr>
  <tr>
    <td>Label</td>
    <td>Estimated label</td>
    <td>Probability</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.90740741</td>
  </tr>
  <tr>
    <td>0<br></td>
    <td>0</td>
    <td>0.90740741</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.91570881</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.90740741</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.89775561</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.89775561</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.67647059</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.91570881</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.91570881</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.67647059</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Gaussian Naive Bayes</th>
  </tr>
  <tr>
    <td>Label</td>
    <td>Estimated label</td>
    <td>Probability</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.999999390</td>
  </tr>
  <tr>
    <td>0<br></td>
    <td>0</td>
    <td>0.999999390</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.569434327</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.841778072</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.835817807</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.835817807</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.612528817</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.883369126</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.831691072</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.521778528</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Support Vector Machine</th>
  </tr>
  <tr>
    <td>Label</td>
    <td>Estimated label</td>
    <td>Probability</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.92314078</td>
  </tr>
  <tr>
    <td>0<br></td>
    <td>0</td>
    <td>0.92694052</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.65021272</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.81390982</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.86003766</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.8605255</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.83425303</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.97584139</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.82403036</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.80041372</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Logistic Regression</th>
  </tr>
  <tr>
    <td>Label</td>
    <td>Estimated label</td>
    <td>Probability</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.79770699</td>
  </tr>
  <tr>
    <td>0<br></td>
    <td>0</td>
    <td>0.78271488</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.56420469</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.56946768</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.86886883</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.8673849</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.5175392</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.66775676</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.55726023</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.52991957</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Voting Classifier</th>
  </tr>
  <tr>
    <td>Label</td>
    <td>Estimated label</td>
    <td>Probability</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.91029576</td>
  </tr>
  <tr>
    <td>0<br></td>
    <td>0</td>
    <td>0.91029576</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.59435627</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.69313934</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.86604043</td>
  </tr>
  <tr>
    <td>0</td>
    <td>0</td>
    <td>0.86604043</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.59705601</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.89258578</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0.81703188</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0.7196419</td>
  </tr>
</table>

### Prerequisites

```
numpy
pandas
scikit-learn
matplotlib
```

### Installing

To install the package, download it from GitHub, and navigate to the
 Task2 folder and install using setup.py
command:

```
$ python setup.py install
```

To use the installed package, you can either import it into your module
with:

```
from ds_internship_task2 import create_new_datasets
from ds_internship_task2 import visualise_dataset
from ds_internship_task2 import titanic_classification
```
Or use the command line script as follows:

```
$ run-all /full/path/to/the/data/folder -cdata -cvis -tmodels -print -splt
```

This will create different datasets (-cdata), create visualisations
(-cvis), train the models (-tmodels), print the results and current
statuses (-print), as well as show plots that are created by the -cvis
argument.

Of course, any of these arguments may be omitted but be careful not to
exclude the -tmodels arg if there are no pretrained models available in
the `/Data/Pickles` folder.

## Month 1 Task 3


## Authors

**Amar Civgin**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
