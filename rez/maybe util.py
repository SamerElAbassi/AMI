def results_tfidf_related(model,vectorizer,train_raw,test_raw,train_synt,test_synt,method_name='tfidf'):
    #task A
    RAW=pd.concat([train_raw,test_raw])
    SYNT=pd.concat([train_synt,test_synt])
    DATAFRAME=pd.concat([RAW,SYNT])
    TRAIN_DATAFRAME=pd.concat([train_raw,train_synt])
    TEST_DATAFRAME=pd.concat([test_raw,test_synt])

    X=vectorizer.fit_transform(RAW['clean'])

    #raw
    train_x,test_x=X[:5000],X[5000:]

    #misog
    model.fit(train_x,train_raw['misogynous'])
    result_mis=model.predict(test_x)

    #agr
    model.fit(train_x,train_raw['aggressiveness'])
    result_agr=model.predict(test_x)

    for i,(mis,agr) in enumerate(zip(result_mis,result_agr)):
        if mis==0 and agr==1:
            result_agr[i]=0 #Sunt 4 cazuri cand se intampla
    create_file(method_name+' task A.tsv',result_mis,result_agr)

    
    #B
    print(DATAFRAME['clean'])
    X=vectorizer.fit_transform(DATAFRAME['clean'])
    x_train=vectorizer.transform(TRAIN_DATAFRAME['clean'])
    x_raw_test=vectorizer.transform(test_raw['clean'])
    x_synt_test=vectorizer.transform(test_synt['clean'])
    model.fit(x_train,TRAIN_DATAFRAME['misogynous'])

    print(x_train.shape)
    print(x_raw_test.shape)
    result_raw=model.predict(x_raw_test)
    create_file_2(method_name+' task B raw.tsv',result_raw,'raw')

    result_synt=model.predict(x_synt_test)
    create_file_2(method_name+' task B synt.tsv',result_synt,'synt')

