import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt

def main():

    st.title("From Suspicion to Confidence: Instantly Classify Your Mushrooms with Our Web App! 🍄")
    st.subheader("Completed by Ghadah AlHabib!")
    st.sidebar.title("Test the functionality of the Web App!")
    st.sidebar.markdown("🍄 Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('/Users/ghadahalhabib/Desktop/WebApp/mushrooms.csv')
        label = LabelEncoder()
        # encode target labels with values between 0 and n_classes-1
        for col in data.columns:
            # fits the encoder to the labels and returns the encoded labels
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df):
        target = df.type
        predictor = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(predictor, target, test_size=0.25, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, y_test, predictions):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
            fig, ax = plt.subplots()
            display.plot(ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, thresholds = precision_recall_curve(y_test, predictions)
            display = PrecisionRecallDisplay(precision=precision, recall=recall)
            fig, ax = plt.subplots()
            display.plot(ax=ax)
            st.pyplot(fig)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose classifier!")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine(SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine(SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, y_test, y_pred)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C_LR = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C_LR, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, y_test, y_pred)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = eval(st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap'))

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, y_test, y_pred)


if __name__ == '__main__':
    main()
