import pickle
def prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title):
    x = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    random_forest = pickle.load(open('titanic/titanic_model.sav','rb'))
    predictions = random_forest.predict(x)
    if predictions == 1:
        return "Survived"
    elif predictions == 0:
        return "Now is with God"
    return "Error"