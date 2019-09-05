# logreg_winperc.py
# Logistic Regression using Win Percentage
# Written by KAC
# Last editted: 08/11/2019

# Load packages for Logistic Regression
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Import Data
from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))
data_dir = '/Users/Koby/PycharmProjects/NCAACompetition/Input/'
df_tour = pd.read_csv('/Users/Koby/PycharmProjects/NCAACompetition/Input/DataFiles/NCAATourneyCompactResults.csv')
df_regseason = pd.read_csv('/Users/Koby/PycharmProjects/NCAACompetition/Input/DataFiles/RegularSeasonCompactResults.csv')

# Finds the record of each team in each year and calculates a winning percentage, merges them all into one data
# frame called df_record
df_regseason.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_wins = pd.DataFrame(df_regseason.groupby(['Season', 'WTeamID']).size().reset_index())
df_wins.rename(columns={df_wins.columns[1]: 'TeamID', df_wins.columns[2]: 'Wins'}, inplace=True)
df_losses = pd.DataFrame(df_regseason.groupby(['Season', 'LTeamID']).size().reset_index())
df_losses.rename(columns={df_losses.columns[1]: 'TeamID', df_losses.columns[2]: 'Losses'}, inplace=True)
df_record = pd.merge(left=df_wins, right=df_losses, how='left', on=['Season', 'TeamID'])
df_record.Losses.fillna(0, inplace=True)  # Gives 0 losses to the teams that finished the regular season undefeated.
df_record['WinPerc'] = df_record.Wins / (df_record.Wins + df_record.Losses)
df_record.drop(labels=['Wins', 'Losses'], inplace=True, axis=1)

# Matches the win percentages of each team with the tournament match up and then calculates a difference

df_temp = df_record.copy()
df_temp.rename(columns={df_temp.columns[1]: 'WTeamID'}, inplace=True)
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_dummy = pd.merge(left=df_tour, right=df_temp, how='left', on=['Season', 'WTeamID'])
df_dummy.rename(columns={df_dummy.columns[3]: 'WTeamPerc'}, inplace=True)
df_temp.rename(columns={df_temp.columns[1]: 'LTeamID'}, inplace=True)
df_dummy = pd.merge(left=df_dummy, right=df_temp, how='left', on=['Season', 'LTeamID'])
df_dummy.rename(columns={df_dummy.columns[4]: 'LTeamPerc'}, inplace=True)
df_dummy['WinPercDiff'] = df_dummy.WTeamPerc - df_dummy.LTeamPerc

# idx, idy = np.where(pd.isnull(df_dummy))
# result = np.column_stack([df_dummy.index[idx], df_dummy.columns[idy]])
# print(result)

# Creates a data frame that summarizes wins and losses & seed differences
df_win = pd.DataFrame()
df_win['WinPercDiff'] = df_dummy['WinPercDiff']
df_win['Result'] = 1

df_loss = pd.DataFrame()
df_loss['WinPercDiff'] = -df_dummy['WinPercDiff']
df_loss['Result'] = 0
df_predictions = pd.concat((df_win, df_loss))

X_train = df_predictions.WinPercDiff.values.reshape(-1, 1)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)

# Creates logistic regression model with different values of C
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)

clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

# Plot of model
X = np.arange(-2, 4).reshape(-1, 1)
preds = clf.predict_proba(X)[:, 1]

plt.plot(X, preds)
plt.xlabel('Team1 Win Perc - Team2 Win Perc')
plt.ylabel('P(Team1 will win)')
plt.show()

# Creating X-test for the model to make predictions with
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_winpct = df_record[(df_record.TeamID == t1) & (df_record.Season == year)].WinPerc.values[0]
    t2_winpct = df_record[(df_record.TeamID == t2) & (df_record.Season == year)].WinPerc.values[0]
    diff_winpct = t1_winpct - t2_winpct
    X_test[ii, 0] = diff_winpct

# Makes predictions using model
preds = clf.predict_proba(X_test)[:, 1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
# print(df_sample_sub.head())

# Creates submission file with the original regression model, no guesses
df_sample_sub.to_csv('2019_Predicitions_logreg_winpct-v1.csv', index=False)


