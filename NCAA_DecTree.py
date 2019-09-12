# NCAA-regression-tree-v1.py
# Regression tree model for Google NCAA competition
# Created: 08/27/2019

# Variables to be offered to the model:
# - Wins    - Losses    - Win%
# - Points per game     - Points against per game
# - Record in games where score diff <= 10 points
# - Seed

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load packages for Logistic Regression
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Import Data
data_dir = '/Users/Koby/PycharmProjects/NCAACompetition/Input/'
df_seeds = pd.read_csv(data_dir + 'DataFiles/NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'DataFiles/NCAATourneyCompactResults.csv')
df_regseason = pd.read_csv(data_dir + 'DataFiles/RegularSeasonCompactResults.csv')


# Get all data from the 'Regular Season Compact Results' file and add it to the df_basedata dataframe
# Record calculations:
df_basedata = pd.DataFrame(df_regseason.groupby(['Season', 'WTeamID']).size().reset_index())
df_basedata.rename(columns={df_basedata.columns[1]: 'TeamID', df_basedata.columns[2]: 'Wins'}, inplace=True)
df_temp = pd.DataFrame(df_regseason.groupby(['Season', 'LTeamID']).size().reset_index())
df_temp.rename(columns={df_temp.columns[1]: 'TeamID', df_temp.columns[2]: 'Losses'}, inplace=True)
df_basedata = pd.merge(left=df_basedata, right=df_temp, how='left', on=['Season', 'TeamID'])
df_basedata.Wins.fillna(0, inplace=True)
df_basedata.Losses.fillna(0, inplace=True)
df_basedata['Win_Perc'] = df_basedata.Wins / (df_basedata.Wins + df_basedata.Losses)
# PPG calculations:
df_ppg = pd.DataFrame(df_regseason.groupby(['Season', 'WTeamID'])['WScore'].sum().reset_index())
df_ppg.rename(columns={df_ppg.columns[1]: 'TeamID', df_ppg.columns[2]: 'WPoints'}, inplace=True)
df_temp2 = pd.DataFrame(df_regseason.groupby(['Season', 'LTeamID'])['LScore'].sum().reset_index())
df_temp2.rename(columns={df_temp2.columns[1]: 'TeamID', df_temp2.columns[2]: 'LPoints'}, inplace=True)
df_ppg = pd.merge(left=df_ppg, right=df_temp2, how='left', on=['Season', 'TeamID'])
df_ppg['Points'] = df_ppg.WPoints + df_ppg.LPoints
df_ppg.Points.fillna(0, inplace=True)
df_ppg['PPG'] = df_ppg.Points / (df_basedata.Wins + df_basedata.Losses)
df_ppg.drop(labels=['WPoints', 'LPoints', 'Points'], inplace=True, axis=1)
df_basedata = pd.merge(left=df_basedata, right=df_ppg, how='left', on=['Season', 'TeamID'])
df_basedata.PPG.fillna(0, inplace=True)
del df_temp
del df_temp2
del df_ppg
# PAPG calculations:
df_papg = pd.DataFrame(df_regseason.groupby(['Season', 'WTeamID'])['LScore'].sum().reset_index())
df_papg.rename(columns={df_papg.columns[1]: 'TeamID', df_papg.columns[2]: 'WPointsAgainst'}, inplace=True)
df_temp2 = pd.DataFrame(df_regseason.groupby(['Season', 'LTeamID'])['WScore'].sum().reset_index())
df_temp2.rename(columns={df_temp2.columns[1]: 'TeamID', df_temp2.columns[2]: 'LPointsAgainst'}, inplace=True)
df_papg = pd.merge(left=df_papg, right=df_temp2, how='left', on=['Season', 'TeamID'])
df_papg['Points_Against'] = df_papg.WPointsAgainst + df_papg.LPointsAgainst
df_papg.Points_Against.fillna(0, inplace=True)
df_papg['PAPG'] = df_papg.Points_Against / (df_basedata.Wins + df_basedata.Losses)
df_papg.drop(labels=['WPointsAgainst', 'LPointsAgainst', 'Points_Against'], inplace=True, axis=1)
df_basedata = pd.merge(left=df_basedata, right=df_papg, how='left', on=['Season', 'TeamID'])
df_basedata.PAPG.fillna(0, inplace=True)
del df_temp2
del df_papg
# Close Games Record Calculations:
df_regseason['score_diff'] = df_regseason.WScore - df_regseason.LScore
df_regseason_close = pd.DataFrame(df_regseason.loc[df_regseason['score_diff'] <= 10])
df_close_record = pd.DataFrame(df_regseason_close.groupby(['Season', 'WTeamID']).size().reset_index())
df_close_record.rename(columns={df_close_record.columns[1]: 'TeamID', df_close_record.columns[2]: 'Close_Wins'},
                       inplace=True)
df_close_record.Close_Wins.fillna(0, inplace=True)
df_temp = pd.DataFrame(df_regseason_close.groupby(['Season', 'LTeamID']).size().reset_index())
df_temp.rename(columns={df_temp.columns[1]: 'TeamID', df_temp.columns[2]: 'Close_Losses'}, inplace=True)
df_close_record = pd.merge(left=df_close_record, right=df_temp, how='left', on=['Season', 'TeamID'])
df_close_record.Close_Losses.fillna(0, inplace=True)
df_basedata = pd.merge(left=df_basedata, right=df_close_record, how='left', on=['Season', 'TeamID'])
df_basedata.Close_Wins.fillna(0, inplace=True)
df_basedata.Close_Losses.fillna(0, inplace=True)
df_basedata['Close_Win_Perc'] = df_basedata.Close_Wins / (df_basedata.Close_Wins + df_basedata.Close_Losses)
df_basedata.Close_Win_Perc.fillna(0, inplace=True)
del df_close_record
del df_regseason_close
del df_temp

# Check to see if there are any empty values
# idx, idy = np.where(pd.isnull(df_basedata))
# result = np.column_stack([df_basedata.index[idx], df_basedata.columns[idy]])
# print(result)

# Gets Tournament team IDs and two temp dataframes with the metrics for each team
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tempWin = df_basedata.copy()
df_tempWin = df_tempWin.rename(columns={'TeamID': 'WTeamID'})
df_tempLose = df_basedata.copy()
df_tempLose = df_tempLose.rename(columns={'TeamID': 'LTeamID'})

# Calculate differences for each metric based on each match up in the tournament
df_tour = pd.merge(left=df_tour, right=df_tempWin, how='left', on=['Season', 'WTeamID'])
df_tour = pd.merge(left=df_tour, right=df_tempLose, how='left', on=['Season', 'LTeamID'])
df_tour['Wins_x'] = df_tour['Wins_x'].sub(df_tour['Wins_y'])
df_tour['Losses_x'] = df_tour['Losses_x'].sub(df_tour['Losses_y'])
df_tour['Win_Perc_x'] = df_tour['Win_Perc_x'].sub(df_tour['Win_Perc_y'])
df_tour['PPG_x'] = df_tour['PPG_x'].sub(df_tour['PPG_y'])
df_tour['PAPG_x'] = df_tour['PAPG_x'].sub(df_tour['PAPG_y'])
df_tour['Close_Wins_x'] = df_tour['Close_Wins_x'].sub(df_tour['Close_Wins_y'])
df_tour['Close_Losses_x'] = df_tour['Close_Losses_x'].sub(df_tour['Close_Losses_y'])
df_tour['Close_Win_Perc_x'] = df_tour['Close_Win_Perc_x'].sub(df_tour['Close_Win_Perc_y'])

## Get seed differences and add them to the comparison
def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)

# Merges seeds for each team, calculates the difference
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'}) # Copy of ID and seed to use for winners
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_tour = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_tour = pd.merge(left=df_tour, right=df_lossseeds, how='left', on=['Season', 'LTeamID'])
df_tour['Seed_Diff'] = df_tour['WSeed'].sub(df_tour['LSeed'])

# Drop unneeded columns from df_tour
df_tour.drop(labels=['Wins_y', 'Losses_y', 'Win_Perc_y', 'PPG_y', 'PAPG_y', 'Close_Wins_y', 'Close_Losses_y',
                     'Close_Win_Perc_y', 'WSeed', 'LSeed'], inplace=True, axis=1)

# Separate Winning and Losing Teams
df_win = pd.DataFrame()
df_win = df_tour.copy()
df_win.drop(labels=['Season', 'WTeamID', 'LTeamID'], inplace=True, axis=1)
df_win['Result'] = 0

df_lose = df_win.copy()
df_lose = df_lose.multiply(-1)
df_lose['Result'] = 1
df_predictions = pd.concat((df_win, df_lose))

# Create training data
X_train = df_predictions.copy()
X_train.drop(labels=['Result'], inplace=True, axis=1)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)

def treeMaker(X, Y, max_depth, min_split, min_leaf):
    ''' The function creates a decision tree classifier and calulates the cross validation score using log loss.
    Returns the score and the values used. Useful to loop over this function to try many parameters.'''
    dec_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=0)
    clf = dec_tree.fit(X, Y)
    score = cross_val_score(dec_tree, X, Y, scoring='neg_log_loss')
    scr = np.mean(score)
    return scr, max_depth, min_split, min_leaf

##Defines the ranges of parameters to try in the treeMaker function
min_samples_splits = np.linspace(0.1, 0.5, 10, endpoint=True)
max_depths = np.linspace(2, 8, 7, endpoint=True)
min_samples_leafs = np.linspace(0.05, 0.3, 10, endpoint=True)

##Loops over the parameter ranges keeping the tree that performs best on log loss scoring
temp = -1
for max_depth in max_depths:
    for min_sample_split in min_samples_splits:
        for min_samples_leaf in min_samples_leafs:
            out = treeMaker(X_train, y_train, max_depth, min_sample_split, min_samples_leaf)
            if out[0] > temp:
                temp = out[0]
                depth = round(out[1], 0)
                split = round(out[2], 2)
                leaf = round(out[3], 2)
                print(out)

# Makes the actual decision tree for predictions. Can use depth, split, and leaf from the code above or be hardcoded.
dec_tree = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf, random_state=0)
clf = dec_tree.fit(X_train, y_train)
score = cross_val_score(dec_tree, X_train, y_train, scoring='neg_log_loss')
scr = np.mean(score)

# VISUALIZE TREE and save as png.
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                feature_names=['Wins', 'Losses', 'Win_Perc', 'PPG', 'PAPG', 'Close_Wins', 'Close_Losses',
                               'Close_Win_Perc', 'Seed_Diff'],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_pdf("tree-v2.pdf")

# Pulls in test cases for 2019 season
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
n_test_games = len(df_sample_sub)

# Gathers the needed data for each match up in 2019
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

X_test = np.zeros(shape=(n_test_games, 9))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_data = df_basedata[(df_basedata.TeamID == t1) & (df_basedata.Season == year)]
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)]
    t1_data = pd.merge(left=t1_data, right=t1_seed, how='left', on=['Season', 'TeamID'])
    t1_data.drop(labels=['Season', 'TeamID'], inplace=True, axis=1)
    t2_data = df_basedata[(df_basedata.TeamID == t2) & (df_basedata.Season == year)]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)]
    t2_data = pd.merge(left=t2_data, right=t2_seed, how='left', on=['Season', 'TeamID'])
    t2_data.drop(labels=['Season', 'TeamID'], inplace=True, axis=1)
    diff_data = t1_data.subtract(t2_data.values)
    X_test[ii] = diff_data

# Makes predictions, clips, and writes to csv.
preds = clf.predict_proba(X_test)
clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.to_csv('2019_Predictions_RegTree-v3.csv', index=False)


