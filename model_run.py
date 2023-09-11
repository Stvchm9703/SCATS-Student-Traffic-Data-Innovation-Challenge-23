# resume
import pandas

temp_df1 = pandas.read_parquet("data/pretrain-snapshot.snappy.parquet")


# model training
# import sklearn


# here to roughly split the data


import numpy
import seaborn
import matplotlib.pyplot as plt

import joblib
import pandas
import time

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



def model_result_rend(modeL_name, y_test, y_pred_lr, auc, fpr, tpr):
    plt.figure()
    group_names = ["True Neg", "False Neg", "False pos", "True Pos"]
    group_counts = [
        "{0:0.0f}".format(value)
        for value in confusion_matrix(y_test, y_pred_lr).flatten()
    ]
    group_percentages = [
        "{0:.2%}".format(value)
        for value in confusion_matrix(y_test, y_pred_lr).flatten()
        / numpy.sum(confusion_matrix(y_test, y_pred_lr))
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = numpy.asarray(labels).reshape(2, 2)
    seaborn.heatmap(
        confusion_matrix(y_test, y_pred_lr), annot=labels, fmt="", cmap="Blues"
    )
    plt.title("{modeL_name} result".format(modeL_name=modeL_name))
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    plt.figure()
    lw = 2  # the line width
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc
    )
    # The dashed line for random choice
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.show()


def generate_train_test_set(x_col, y_col, new_edited_trainset, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        new_edited_trainset[x_col],
        new_edited_trainset[y_col],
        test_size=0.3,
        random_state=random_state,
    )
    return [X_train, X_test, y_train, y_test]



def model_randomForstClassifier(
    X_train: pandas.DataFrame,
    X_test: pandas.DataFrame,
    y_train: pandas.DataFrame,
    y_test: pandas.DataFrame,
):
    timer = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf.fit(X_train, y_train)
    _score = clf.score(X_test, y_test)
    y_probs = clf.predict_proba(X_test)
    # print(y_probs)
    # y_probs_list = [ int(b > a) for ([a, b]) in y_probs ]
    y_probs_list = y_probs
    roc = roc_auc_score(
        y_test[y_test.columns[0]].to_list(),
        y_probs_list,
        multi_class="ovr",
        average="weighted",
    )
    fpr, tpr, thresholds = roc_curve(
        y_test[y_test.columns[0]].to_list(), y_probs_list, pos_label=1
    )

    print(classification_report(y_test[y_test.columns[0]].to_list(), y_probs_list))

    # roc_list.append(_score)
    print(
        "score in test-set for {model}: {_score}, roc: {roc} ".format(
            model=clf.__class__.__name__, _score=_score, roc=roc
        )
    )
    new_timer = time.perf_counter()
    time_diff = new_timer - timer
    model_result_rend(
        modeL_name=clf.__class__.__name__,
        y_test=y_test,
        y_pred_lr=y_probs_list,
        auc=roc,
        fpr=fpr,
        tpr=tpr,
    )

    joblib.dump(
        clf,
        "./data/model_{model_name}_{start_time}.pkl".format(
            model_name=clf.__class__.__name__, start_time=timer
        ),
    )
    return {
        # 'model': clf,
        "model_name": clf.__class__.__name__,
        "score": _score,
        "roc_auc": roc,
        "complete_time": time_diff,
        "start_time": timer,
        "end_time": new_timer,
    }


# Method 3 : MLP Classifier
def model_mlp_classifier(X_train: pandas.DataFrame, X_test: pandas.DataFrame, y_train: pandas.DataFrame, y_test: pandas.DataFrame):
    timer = time.perf_counter()
    clf = MLPClassifier(
        hidden_layer_sizes=(18, 10, 6, 2),
        max_iter=15000)
    # print('inited model, start fitting')
    clf.fit(X_train, y_train[y_train.columns[0]].to_list())
    # print('fitted, start score')

    _score = clf.score(X_test, y_test)

    # Get the probabilities of each class.
    y_probs = clf.predict_proba(X_test)
    # y_probs_list = [
    #     int(b > a) for ([a, b]) in y_probs
    # ]
    y_probs_list = y_probs
    roc = roc_auc_score(
        y_test[y_test.columns[0]].to_list(),
        y_probs_list,
        multi_class="ovr", average="weighted")
    fpr, tpr, thresholds = roc_curve(y_test, y_probs_list, pos_label=1)
    # roc_list.append(_score)
    print(classification_report(
        y_test[y_test.columns[0]].to_list(), y_probs_list))
    print("score in test-set for {model}: {_score}, roc: {roc} ".format(
        model=clf.__class__.__name__, _score=_score, roc=roc))
    new_timer = time.perf_counter()
    time_diff = (new_timer - timer)
    model_result_rend(modeL_name=clf.__class__.__name__, y_test=y_test,
                      y_pred_lr=y_probs_list, auc=roc, fpr=fpr, tpr=tpr)

    joblib.dump(clf, './data/model_{model_name}_{start_time}.pkl'.format(
        model_name=clf.__class__.__name__, start_time=timer))
    return {
        # 'model': clf,
        'model_name': clf.__class__.__name__,
        'score': _score,
        'roc_auc': roc,
        'complete_time': time_diff,
        'start_time': timer,
        'end_time': new_timer,
    }



target_result = ["duration"]
params = temp_df1.columns.to_list()
params = [x for x in params if x not in target_result]

[X_train, X_test, y_train, y_test] = generate_train_test_set(
    x_col=params,
    y_col=target_result,
    new_edited_trainset=temp_df1,
    random_state=7,
)



model_mlp_classifier(X_train, X_test, y_train, y_test)