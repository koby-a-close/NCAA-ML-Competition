# NCAA-ML-Competition
Project Background and Goals:
This project is hosted by Google and the NCAA. The data provided by Google includes regular season and tournament data from 1984 on.

The goal is to predict the outcomes of the 2019 Men's NCAA Tournament using historical data to build a model. Since we don't know who will win each game, every possible combination is predicted by the model and the relvant match ups are pulled during scoring. Submissions are scored using log loss. This method of scoring penalizes you heavily for being both confident and wrong.

Personal Goals:
Since this project was my first I wanted to set some goals for myself:
1) Create a model that outperformed the Google Starter Kernel
        I set this goals after finshing the Google Starter Kernel (discussed more below) which had a log loss score of               0.51455 and would have placed 508th.
2) Create a model that outperformed the picks I used for bracket competitions. 
        My family does a bracket competition where winners are chosen round by round so this is a fair comparison. My final           record in 2019 was 42-24.
        
Skills Used/Added from this Project:
- Python Packages: pandas, numpy, sklearn, xgboost
- Model Creation: logistic regression, decision trees, gradient boosting, feature selection, cross validation, performance                        evaluation
- Data wrangling 

Models Built:
-Google Starter Kernel (Logistic Regression using Seed), file: Google_Starter_Kernel.py
-Google Starter Kernel with random guessing strategy, file: Google_Starter_Kernel.py 
-Logistic Regression using Winning Percentage, file: LogReg_WinPerc.py
-Decision Tree using Wins, Losses, Win %, Points per Game (PPG), Points Against per Game (PAPG), Wins in Close Games, Losses          in Close Games, Win % in Close Games, file: NCAA_DecTree.py
-XGBoost Ensemble Model, file: NCAA_xgboost.py

Results:
My best log loss score was 0.48970 from the Google Starter Kernel with Guessing Strategy model and it would have placed 304th. It's record was 47-19. However, it relied on guessing. My best log loss score with a model only was 0.50034 from the XGBoost model and it would have placed 420th. It's record was 44-22. I met both of my goals.

*** Please see the NCAA-Kaggle-Report pdf for a full report on my process, results, and conclusions. ***
