





fullset21 = pd.read_pickle('D:/MSc_project/datapickles/fullset_21.pkl')

grid_mcc_xgb = []
grid_params_xgb = []

df = fullset21

X = df.drop(['posmiddlePRO', 'omegaPRO', 'cistrans'], axis=1)
Xcols = [c for c in X.columns]
ohe = OneHotEncoder() # creating the onehotencoder for the data
X_enc = pd.DataFrame(ohe.fit_transform(X[Xcols]).toarray())
y = pd.get_dummies(df['cistrans'], drop_first=True).rename(columns={'trans':'cistrans conformation'}) # creating dummies for cis trans conformation using the rename() method
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size= 0.1)

xgb = XGBClassifier(n_estimators=100, tree_method= 'exact', use_label_encoder=False, verbosity=2,objective='binary:logistic')
paramgrid = {'n_estimators': [100]} # performing a grid search to find the best parameters
grid = GridSearchCV(xgb, paramgrid, cv=5, verbose=3) # grid is the model, could be KNN or random forests
grid.fit(X_train, y_train.values.ravel()) # fitting the model
xgb_predictions = grid.predict(X_test) # getting the preds

filename_rfc = 'D:/MSc_project/func_testing/new_sets/models/ohe_rfc_df21_single_cv5.sav'
joblib.dump(grid, filename_rfc)

grid_mcc_xgb.append(metrics.matthews_corrcoef(y_test, xgb_predictions).round(3))
grid_params_xgb.append(grid.best_params_)
print(metrics.plot_confusion_matrix(grid,X_test,y_test, display_labels=['cis', 'trans'], cmap=plt.cm.Blues, normalize='true'))
print(metrics.plot_roc_curve(grid, X_test, y_test))
