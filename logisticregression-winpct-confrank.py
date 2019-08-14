# logisticregression-winpct-confrank.py
# Logistic Regression using Win Percentage and Conference Ranking
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
data_dir = '../input/'
df_tour = pd.read_csv(data_dir + 'datafiles/NCAATourneyCompactResults.csv')
df_regseason = pd.read_csv(data_dir + 'datafiles/RegularSeasonCompactResults.csv')

# Finds the record of each team in each year and calculates a winning percentage, merges them all into one data
# frame called df_record
df_regseason.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_wins = pd.DataFrame(df_regseason.groupby(['Season', 'WTeamID']).size().reset_index())
df_wins.rename(columns={df_wins.columns[1]: 'Team ID',df_wins.columns[2]: 'Wins'}, inplace=True)
df_losses = pd.DataFrame(df_regseason.groupby(['Season', 'LTeamID']).size().reset_index())
df_losses.rename(columns={df_losses.columns[1]: 'Team ID',df_losses.columns[2]: 'Losses'}, inplace=True)
df_record = pd.merge(left=df_wins, right=df_losses, how='left', on=['Season','Team ID'])
df_record.Losses.fillna(0, inplace=True)
df_record['Win Percentage'] = df_record.Wins/(df_record.Wins+df_record.Losses)

df_record.drop(labels=['Wins', 'Losses'], inplace=True, axis=1)
df_record.rename(columns={df_record.columns[1]: 'WTeamID'}, inplace=True)

# Matches the win percentages of each team with the tournament match up and then calculates a difference
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_dummy = pd.merge(left=df_tour, right=df_record, how='left', on=['Season','WTeamID'])
df_dummy.rename(columns={df_dummy.columns[3]: 'WTeamPerc'}, inplace=True)
df_record.rename(columns={df_record.columns[1]: 'LTeamID'}, inplace=True)
df_dummy = pd.merge(left=df_dummy, right=df_record, how='left', on=['Season','LTeamID'])
df_dummy.rename(columns={df_dummy.columns[4]: 'LTeamPerc'}, inplace=True)
df_dummy['WinPercDiff'] = df_dummy.WTeamPerc - df_dummy.LTeamPerc

#idx, idy = np.where(pd.isnull(df_dummy))
#result = np.column_stack([df_dummy.index[idx], df_dummy.columns[idy]])
#print(result)

# Creates a data frame that summarizes wins and losses & seed differences
df_win = pd.DataFrame()
df_win['WinPercDiff'] = df_dummy['WinPercDiff']
df_win['Result'] = 1

df_loss = pd.DataFrame()
df_loss['WinPercDiff'] = -df_dummy['WinPercDiff']
df_loss['Result'] = 0
df_predictions = pd.concat((df_win, df_loss))

X_train = df_predictions.WinPercDiff.values.reshape(-1,1)
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
preds = clf.predict_proba(X)[:,1]

plt.plot(X, preds)
plt.xlabel('Team1 Win Perc - Team2 Win Perc')
plt.ylabel('P(Team1 will win)')
plt.show()



