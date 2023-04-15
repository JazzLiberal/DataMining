{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMCjwZGzXTizQlmKOERLxEw",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JazzLiberal/DataMining/blob/main/FinalModel122108729\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fwBULCkKMDrC",
        "outputId": "8c51d1a0-c118-4335-92be-93c967f74c01"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The best classifier is: Pipeline(steps=[('scaler', MinMaxScaler()),\n",
            "                ('pca', PCA(n_components=24, whiten=True)),\n",
            "                ('rf',\n",
            "                 RandomForestClassifier(max_features=0.3, min_samples_leaf=2,\n",
            "                                        n_estimators=9))])\n",
            "Best hyper-parameters:  {'pca__n_components': 24, 'pca__whiten': True, 'rf__max_depth': None, 'rf__max_features': 0.3, 'rf__min_samples_leaf': 2, 'rf__n_estimators': 9}\n",
            "Best score:  0.9223063973063973\n",
            "Accuracy score:  0.9417989417989417\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn import model_selection\n",
        "from sklearn import preprocessing\n",
        "from sklearn import tree\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "\n",
        "\n",
        "df = pd.read_csv(\"https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/test_dataset.csv\", index_col=0)\n",
        "df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)\n",
        "\n",
        "\n",
        "X = df.iloc[:,:-1]\n",
        "y = df['target']\n",
        "\n",
        "encoder = preprocessing.LabelEncoder()\n",
        "y = encoder.fit_transform(y)\n",
        "\n",
        "#Data Split as before\n",
        "\n",
        "train_features, test_features, train_labels, test_labels = model_selection.train_test_split(X,y, test_size=0.3, random_state=999)\n",
        "\n",
        "#Define a new Pipeline using the RandomForest Classifier \n",
        "pipe = Pipeline([('scaler', MinMaxScaler()), ('pca', PCA()), ('rf', RandomForestClassifier())])\n",
        "\n",
        "\n",
        "#update parameter grid for new random forest and pca arguments\n",
        "param_grid = {'pca__n_components': [24],\n",
        "              'pca__whiten' : [True],\n",
        "              'rf__max_features': [0.3],\n",
        "              'rf__max_depth': [None],\n",
        "              'rf__min_samples_leaf' : [2],\n",
        "              'rf__n_estimators': [9]}\n",
        "\n",
        "\n",
        "#Define and fit the grid as before, trialled different cross validations\n",
        "grid = GridSearchCV(pipe, param_grid, cv=8)\n",
        "grid.fit(train_features, train_labels)\n",
        "\n",
        "print(\"Best score: \", grid.best_score_)\n",
        "\n",
        "y_pred = grid.best_estimator_.predict(test_features)\n",
        "\n",
        "# Compute the accuracy score of the model\n",
        "accuracy = accuracy_score(test_labels, y_pred)\n",
        "print(\"Accuracy score: \", accuracy)    "
      ]
    }
  ]
}
