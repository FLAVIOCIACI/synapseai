import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st
from math import sqrt
import numpy as np
import time
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder


sns.set_style("darkgrid")


st.title("SYNAPSE(AI)")


st.markdown("""


""")
st.sidebar.image('logo/logo.png', caption='Synapse(AI) Tools')
st.sidebar.title("CARICA DATASET")
st.sidebar.write('<p class="big-font">Dataset offLine</p>',unsafe_allow_html=True)
caricaOffLine=st.sidebar.checkbox("Carica Dataset OffLine",False)
st.sidebar.write('<p class="big-font">Dataset OnLine</p>',unsafe_allow_html=True)
caricaOnLine=st.sidebar.checkbox("Carica Dataset OnLine",False)
st.sidebar.title("DATA ANALYS")

#st.subheader("Checkbox")
w1 = st.sidebar.checkbox("Data", False)
st.sidebar.title("GRAFICI")
pairplot=st.sidebar.checkbox("Pair Plot Studio delle relazioni tra variabili",False)
plot= st.sidebar.checkbox("Grafico di dispersione", False)
plothist= st.sidebar.checkbox("show hist plots", False)

scartermatrix= st.sidebar.checkbox("Scatter Matrix", False)
#distView=st.sidebar.checkbox("Dist View", False)
#_3dplot=st.sidebar.checkbox("3D plots", False)
linechart=st.sidebar.checkbox("Linechart",False)
QQplot=st.sidebar.checkbox("QQplot",False)
st.sidebar.title("CONFRONTO MODELLI PREDETTIVI")
classificatore=st.sidebar.checkbox("Classificatore Modelli",False)
st.sidebar.title("Elabora singoli MODELLI PREDETTIVI")
trainmodelLinearRegression= st.sidebar.checkbox("Train model LinearRegression", False)
trainmodelKNeighborsClassifier= st.sidebar.checkbox("Train model KNeighborsClassifier", False)
trainmodelMLPClassifier= st.sidebar.checkbox("Train model MLPClassifier", False)
randomClassifier= st.sidebar.checkbox("randomClassifier", False)
dokfold= st.sidebar.checkbox("DO KFold", False)
MultifeatureScatterPlot=st.sidebar.checkbox("Multifeature Scatter Plot",False)


#st.write(w1)
try:
    if caricaOnLine:
        uploaded_file = st.text_input('inserisci il percorso')
        st.write('inserisci il percorso ', uploaded_file)
        data=pd.read_csv(uploaded_file, index_col=0)
        #st.write(list(data))
        colonne=(list(data))
        colonneY=(list(data))
        # nascondi=st.empty()
        # if st.button('Carica il Dataset OnLine:'):
            # data=pd.read_csv(uploaded_file, index_col=0)
            # #st.write(list(data))
            # colonne=(list(data))
            # colonneY=(list(data))
        
                
    if caricaOffLine:
        def read_data():
            uploaded_file = st.file_uploader("Carica Dataset da elaborare...", type=['csv'])

            if uploaded_file is not None:
                
            
                return pd.read_csv(uploaded_file, index_col=0)
        # nascondi=st.empty()

        data=read_data()
        #st.write(list(data))
        colonne=(list(data))
        colonneY=(list(data))
        ##st.write(data)

        # st.write(selcolsX)
        # st.write(selcolsY)
        if pairplot:
            st.subheader("Pair Plot Studio delle relazioni tra variabili")
            try:
                import seaborn as sns
                st.dataframe(data,width=2000,height=500)
                
                fig = sns.pairplot(data)
                st.pyplot(fig)
            except:
                st.error("Carica il Dataset")
                st.stop()
    if w1:
        import seaborn as sns
        import plotly.express as px
        st.subheader("Comprensione del quadro generale")
        
        st.dataframe(data,width=2000,height=500)
        #print(data.head())
        st.subheader("numero arbitrario di righe")
        st.write('<p class="big-font">', data.head(), '</p>',unsafe_allow_html=True)
        st.subheader("distribuzione del dataset")
        st.write('<p class="big-font">', data.describe(), '</p>',unsafe_allow_html=True)
        st.subheader("Numero righe duplicate")
        st.write('<p class="big-font">', data.duplicated().sum(), '</p>',unsafe_allow_html=True)
     
        #print(data.shape)
        #st.write('<p class="big-font">', data.shape, '</p>',unsafe_allow_html=True)
        fig = px.imshow(data, text_auto=True, aspect="auto")

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)


    if randomClassifier:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score
        source = data
        colonnaX1 = list(data)
        selcolsX1 = st.selectbox("select columns", colonnaX1,1)
        colonnaY1 = list(data)
        selcolsY1 = st.selectbox("select columns", colonnaY1,2)
        epoca = st.slider('epochs?', 0, 9,0)
        X = selcolsX1
        y = selcolsY1

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

        oversample = RandomOverSampler(sampling_strategy='minority')

        X_over, y_over = oversample.fit_resample(X_train,y_train)


        rf = RandomForestClassifier()
        rf.fit(X_over,y_over)
        st.write(X_over, y_over)
        # %%
        preds = rf.predict(X_test)
        st.write("X_test:",preds)
        predsTrain = rf.predict(X_train)
        st.write("X_train:",predsTrain)
        st.write("Accuratezza:",accuracy_score(y_test,preds))
        #print(rf)
        # %%
        #import joblib 
        #joblib.dump(preds, 'temposala.pkl') 

        # creiamo un LSTM neural network model
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.summary()

        # Alleniamo il modello
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(xtrain, ytrain, batch_size=1, epochs=epoca)
        
    if scartermatrix:
        import altair as alt
        source = data
        colonnaX2 = list(data)
        selcolsX2 = st.selectbox("select columns", colonnaX2,1)
        colonnaY2 = list(data)
        selcolsY2 = st.selectbox("select columns", colonnaY2,2)
        chart = alt.Chart(source).mark_circle().encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
            color='Origin:N'
        ).properties(
            width=600,
            height=600
        ).repeat(
            row=[selcolsX2],
            column=[selcolsY2]
        ).interactive()

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

        with tab1:
            st.altair_chart(chart, theme="streamlit", use_container_width=True)
        with tab2:
            st.altair_chart(chart, theme=None, use_container_width=True)
    if MultifeatureScatterPlot:
        import altair as alt
        source = data
        colonnaX3 = list(data)
        selcolsX3 = st.selectbox("select columns", colonnaX3,1)
        colonnaY3 = list(data)
        selcolsY3 = st.selectbox("select columns", colonnaY3,2)
        chart = alt.Chart(source).mark_circle().encode(
            alt.X(selcolsX3, scale=alt.Scale(zero=False)),
            alt.Y(selcolsY3, scale=alt.Scale(zero=False, padding=1)),
            color=selcolsX3,
            size=selcolsY3
        )

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

        with tab1:
            st.altair_chart(chart, theme="streamlit", use_container_width=True)
        with tab2:
            st.altair_chart(chart, theme=None, use_container_width=True)
            
    if trainmodelMLPClassifier:
        
        st.header("Modeling MLPClassifier")
       
        colonnaX4 = list(data)
        selcolsX4 = st.selectbox("select columns", colonnaX4,1)
        
        X4=data[list(data)]
        y4=data[[selcolsX4]].values
        X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size=0.3)
        t_start = time.time()
        lrgr = MLPClassifier(alpha = 1)
        lrgr.fit(X_train,y_train)
        t_end = time.time()
        t_diff = t_end - t_start
        train_score = lrgr.score(X_train, y_train)
        test_score = lrgr.score(X_test, y_test)
        pred = lrgr.predict(X_test)
        mse = mean_squared_error(y_test,pred)
        rmse = sqrt(mse)

        st.markdown(f"""

        Linear Regression model trained :
            - MSE:{mse}
            - RMSE:{rmse}
            - START:{t_start}
            - TRAIN SCORE:{train_score}
            - test_score:{test_score}
            - END:{t_end}
            - DIFF:{t_diff}
        """)
        st.success('Model trained successfully MLPClassifier')
    if QQplot:
         #Q-Q plot
        dataelenco=list(data)
        colonnaX5 = list(data)
        selcolsX5 = st.selectbox("select columns", colonnaX5,1)
        colonnaY5 = list(data)
        selcolsY5 = st.selectbox("select columns", colonnaY5,2)
        # st.subheader("Distributions of each columns X")
        sel_colsX5 = selcolsX5
        # st.write(sel_colsX)
        # st.subheader("Distributions of each columns Y")
        sel_colsY5 = selcolsY5
        # st.write(sel_colsY)
        Y5 = data[sel_colsY5]
        X5 = data[sel_colsX5]
        
        #Running the model
        model = sm.OLS(Y5, X5, missing='drop')
        model_result = model.fit()
        qqplot=sm.qqplot(model_result.resid, line='s')
        plt.show()
        st.plotly_chart(qqplot)
        #Kitchen sink model
        ks = sm.OLS(Y5, X5)
        ks_res =ks.fit()
        st.write(ks_res.summary())
        st.subheader("Scatterplot")
        sel_colsXx5 = selcolsX5
        st.write("Scatterplot :",sel_colsXx5)
        
        import plotly.express as px
        fig = px.scatter_matrix(data,
        dimensions=[dataelenco],
        color=sel_colsXx5, symbol=sel_colsXx5,
        title="Scatter matrix data set",
        labels={col:col.replace('_', ' ') for col in data.columns}) # remove underscore
        
        fig.update_layout(title="Scatter matrix data set",
                      dragmode='select',
                      width=1000,
                      height=1000,
                      hovermode='closest')
        st.plotly_chart(fig)
    if linechart:
        st.subheader("Line chart")
        colonnaX6 = list(data)
        selcolsX6 = st.selectbox("select columns", colonnaX6,1)
        colonnaY6 = list(data)
        selcolsY6 = st.selectbox("select columns", colonnaY6,2)
        st.line_chart(data,columns=[data[selcolsX6],data[selcolsY6]])

    if plothist:
        st.subheader("Distributions of each columns")
        dataelenco7=list(data)
        colonnaX7 = list(data)
        selcolsX7 = st.selectbox("select columns", colonnaX7,1)
            
        sel_cols7 = selcolsX7
        st.write(sel_cols7)
        #f=plt.figure()
        fig = go.Histogram(x=data[sel_cols7],nbinsx=10)
        st.plotly_chart([fig])
        

    #    plt.hist(data[sel_cols])
    #    plt.xlabel(sel_cols)
    #    plt.ylabel("sales")
    #    plt.title(f"{sel_cols} vs Sales")
        #plt.show()	
    #    st.plotly_chart(f)

    if plot:#GRAFICO A DISPERSIONE
        # I grafici a dispersione vengono utilizzati per determinare l'intensità di una relazione tra due variabili numeriche. 
        # L'asse x rappresenta la variabile indipendente e l'asse y rappresenta la variabile dipendente. 
        # I grafici a dispersione possono rispondere a domande sui dati come: Che relazione esiste tra due variabili?
        try:
        
            st.subheader("Grafico di dispersione")
            dataelenco8=list(data)
            colonnaX8 = list(data)
            selcolsX8 = st.selectbox("SELEZIONA VALORI X", colonnaX8,1)
            colonnaY8 = list(data)
            selcolsY8 = st.selectbox("SELEZIONA VALORI Y", colonnaY8,2)
            options8 = selcolsX8
            w7 = options8
            # st.write(w7)
            # st.write(selcolsY)
            f=plt.figure()
            plt.scatter(data[w7],data[selcolsY8])
            plt.xlabel(w7)
            plt.ylabel(selcolsY8)
            plt.title(f"{w7} vs {selcolsY8}")
            
            #plt.show()	
            st.plotly_chart(f)
        except:
            st.error("Valori non validi per elaborare il Grafico a Dispersione !!!")
            st.stop()

    # if distView:
        # st.subheader("Combined distribution viewer")
        # # Add histogram data

        # # Group data together
        # hist_data = [data["male"].values,data["age"].values,data["education"].values,data["currentSmoker"].values]

        # group_labels = ["male","age","education","currentSmoker"]

        # # Create distplot with custom bin_size
        # fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

        # # Plot!
        # st.plotly_chart(fig)

    # if _3dplot:
        # options = st.multiselect(
         # 'Enter columns to plot',('male', 'radio'),('age', 'education', 'currentSmoker', 'cigsPerDay'))
        # st.write('You selected:', options)
        # st.subheader("TV & Radio vs Sales")
        # hist_data = [data["male"].values,data["age"].values,data["education"].values,data["currentSmoker"].values]

        # #x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
        # trace1 = go.Scatter3d(
            # x=hist_data[0],
            # y=hist_data[1],
            # z=data["male"].values,
            # mode="markers",
            # marker=dict(
                # size=8,
                # #color=data['sales'],  # set color to an array/list of desired values
                # colorscale="Viridis",  # choose a colorscale
        # #        opacity=0.,
            # ),
        # )

        # data = [trace1]
        # layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        # fig = go.Figure(data=data, layout=layout)
        # st.write(fig)



    # trainmodel= st.checkbox("Train model", False)

    if trainmodelLinearRegression:
        st.header("Modeling LinearRegression")
        dataelenco9=list(data)
        colonnaX9 = list(data)
        selcolsX9 = st.selectbox("select columns", colonnaX9,1)

           
        y9=data[colonnaX9]
        X9=data[[selcolsX9]].values
        X_train, X_test, y_train, y_test = train_test_split(X9, y9, test_size=0.3)
        t_start = time.time()
        lrgr = LinearRegression()
        lrgr.fit(X_train,y_train)
        t_end = time.time()
        t_diff = t_end - t_start
        train_score = lrgr.score(X_train, y_train)
        test_score = lrgr.score(X_test, y_test)
        pred = lrgr.predict(X_test)
        mse = mean_squared_error(y_test,pred)
        rmse = sqrt(mse)

        st.markdown(f"""

        Linear Regression model trained :
            - MSE:{mse}
            - RMSE:{rmse}
            - START:{t_start}
            - TRAIN SCORE:{train_score}
            - test_score:{test_score}
            - END:{t_end}
            - DIFF:{t_diff}
        """)
        
        # acc = accuracy_score(pred, y_test)
        # st.write('<p class="big-font">ACCURANCY:', acc, '</p>',unsafe_allow_html=True)
        
        
        st.success('Model trained successfully LinearRegression')
        
    if trainmodelKNeighborsClassifier:
        st.header("Modeling KNeighborsClassifier")
        dataelenco10=list(data)
        colonnaX10 = list(data)
        selcolsX10 = st.selectbox("select columns", colonnaX10,1)
        # # y10=data[[selcolsX10]].values
        # # X10=data[[selcolsX10]].values
        # X10 = np.array(data[colonnaX10])
        # y10 = data[selcolsX10]
        # X_train, X_test, y_train, y_test = train_test_split(X10, y10, test_size=0.3)
        # t_start = time.time()
        # lrgr = KNeighborsClassifier()
        # lrgr.fit(X_train,y_train)
        # t_end = time.time()
        # t_diff = t_end - t_start
        # train_score = lrgr.score(X_train, y_train)
        # test_score = lrgr.score(X_test, y_test)
        # pred = lrgr.predict(X_test)
        # mse = mean_squared_error(y_test,pred)
        # rmse = sqrt(mse)

        # st.markdown(f"""

        # Linear Regression model trained :
            # - MSE:{mse}
            # - RMSE:{rmse}
            # - START:{t_start}
            # - TRAIN SCORE:{train_score}
            # - test_score:{test_score}
            # - END:{t_end}
            # - DIFF:{t_diff}
        # """)
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        X = data.drop(selcolsX10,axis=1)
        y = data[selcolsX10]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10)

        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        p_test = model.predict(X_test)
        acc = accuracy_score(y_test, p_test)
        st.write('<p class="big-font">ACCURANCY:', acc, '</p>',unsafe_allow_html=True)
        st.success('Model trained successfully KNeighborsClassifier')
    if dokfold:
        st.subheader("KFOLD Random sampling Evalution")
        st.empty()
        my_bar = st.progress(0)

        from sklearn.model_selection import KFold

        X=data.values[:,-1].reshape(-1,1)
        y=data.values[:,-1]
        #st.progress()
        kf=KFold(n_splits=10)
        #X=X.reshape(-1,1)
        mse_list=[]
        rmse_list=[]
        r2_list=[]
        idx=1
        fig=plt.figure()
        i=0
        for train_index, test_index in kf.split(X):
        #	st.progress()
            my_bar.progress(idx*10)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lrgr = LinearRegression()
            lrgr.fit(X_train,y_train)
            pred = lrgr.predict(X_test)
            
            mse = mean_squared_error(y_test,pred)
            rmse = sqrt(mse)
            r2=r2_score(y_test,pred)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            plt.plot(pred,label=f"dataset-{idx}")
            idx+=1
        plt.legend()
        plt.xlabel("Data points")
        plt.ylabel("PRedictions")
        plt.show()
        st.plotly_chart(fig)

        res=pd.DataFrame(columns=["MSE","RMSE","r2_SCORE"])
        res["MSE"]=mse_list
        res["RMSE"]=rmse_list
        res["r2_SCORE"]=r2_list

        st.write(res)
        st.balloons()
    #st.subheader("results of KFOLD")

    #f=res.plot(kind='box',subplots=True)
    #st.plotly_chart([f])

    if classificatore:
        #2.1 Caricamento, analisi ed elaborazione del set di dati:
        #data = pd.read_csv('framingham.csv')
        #data = data.dropna()
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import tree
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        data = data.dropna()
        
        colonnaX = list(data)
        
        st.write('<p class="big-font">', colonnaX, '</p>',unsafe_allow_html=True)
        selcolsX = st.selectbox("select columns", colonnaX,1)
            
            
        if st.button('Check availability'):
            
            
            #2.1 Caricamento, analisi ed elaborazione del set di dati:
            # dividiamo i dati
            from sklearn.model_selection import train_test_split
            # y = np.array(list(data))
            # x = np.array(selcolsX)
            
            x = np.array(data[colonnaX])
            st.write(x)
            y =  y = data[selcolsX]
            st.write(y)
            X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=0.10)
            dict_classifiers = {
                "Logistic Regression": LogisticRegression(),
                "Nearest Neighbors": KNeighborsClassifier(),
                "Linear SVM": SVC(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
                "Decision Tree": tree.DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=1000),
                "Neural Net": MLPClassifier(alpha = 1),
                "Naive Bayes": GaussianNB(),
                #"AdaBoost": AdaBoostClassifier(),
                #"QDA": QuadraticDiscriminantAnalysis(),
                #"Gaussian Process": GaussianProcessClassifier()
            }


            def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
                """
                This method, takes as input the X, Y matrices of the Train and Test set.
                And fits them on all of the Classifiers specified in the dict_classifier.
                The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
                is because it is very easy to save the whole dictionary with the pickle module.

                Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
                So it is best to train them on a smaller dataset first and
                decide whether you want to comment them out or not based on the test accuracy score.
                """

                dict_models = {}
                for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
                    t_start = time.perf_counter()
                    classifier.fit(X_train, Y_train)
                    t_end = time.perf_counter()

                    t_diff = t_end - t_start
                    train_score = classifier.score(X_train, Y_train)
                    test_score = classifier.score(X_test, Y_test)

                    dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                                    'train_time': t_diff}
                    if verbose:
                        print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
                        
                        st.write('<p class="big-font">', "trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff), '</p>',unsafe_allow_html=True)
                return dict_models


            def display_dict_models(dict_models, sort_by='test_score'):
                cls = [key for key in dict_models.keys()]
                test_s = [dict_models[key]['test_score'] for key in cls]
                training_s = [dict_models[key]['train_score'] for key in cls]
                training_t = [dict_models[key]['train_time'] for key in cls]

                data_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)),
                                   columns=['classifier', 'train_score', 'test_score', 'train_time'])
                for ii in range(0, len(cls)):
                    data_.loc[ii, 'classifier'] = cls[ii]
                    data_.loc[ii, 'train_score'] = training_s[ii]
                    data_.loc[ii, 'test_score'] = test_s[ii]
                    data_.loc[ii, 'train_time'] = training_t[ii]
                print(data_.sort_values(by=sort_by, ascending=False))
                web = data_.sort_values(by=sort_by, ascending=False)
                st.write('<p class="big-font">', web, '</p>',unsafe_allow_html=True)
            dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=8)
            display_dict_models(dict_models)

            #4.1 La matrice di correlazione
            correlation_matrix = data.corr()
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, fmt='.2f', cmap='GnBu',cbar_kws={"shrink": .5}, robust=True)
            plt.title('Correlation matrix between the features', fontsize=20)
            plt.show()

            #4.2 Correlazione di una singola funzionalità con altre caratteristiche
            def display_corr_with_col(data, col):
                correlation_matrix = data.corr()
                correlation_type = correlation_matrix[col].copy()
                abs_correlation_type = correlation_type.apply(lambda x: abs(x))
                desc_corr_values = abs_correlation_type.sort_values(ascending=False)
                y_values = list(desc_corr_values.values)[1:]
                x_values = range(0,len(y_values))
                xlabels = list(desc_corr_values.keys())[1:]
                fig, ax = plt.subplots(figsize=(8,8))
                ax.bar(x_values, y_values)
                ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
                ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
                plt.xticks(x_values, xlabels, rotation='vertical')
                plt.show()
                st.write('<p class="big-font">',  plt.show(), '</p>',unsafe_allow_html=True)
            display_corr_with_col(data, 'Type')
except:
        st.error("Carica il Dataset")
        st.stop()

 