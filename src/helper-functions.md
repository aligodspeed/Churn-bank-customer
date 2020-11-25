Helper function to get cross_val_score

def model_output(model, X_t, X_val, y_t, y_val):
    '''Can be used on final test and train validation''
    input:   model, X_t, X_val, y_t, y_val
    or 
    input:   model, X_train, X_test, y_train, y_test
    '''
    model.fit(X_t, y_t)
    y_hat = model.predict(X_val)
    print(f'''The Cross Val f1 score is: {cross_val_score(estimator = model, X = X_t,y = y_t, cv = 3, scoring = 'f1').mean()}''')
    print(f'The test Accuracy is: {accuracy_score(y_val, y_hat)}')
    print(confusion_matrix(y_val, y_hat))
    print(classification_report(y_val, y_hat))
    return
    
    
    
    
Helper function to make plots


def confusion_matrix_info(model, X_train, y_train, save_path=None):
    '''
    Creates a confusion matrix for a given model
    Parameters
    ----------
    model: an estimator
    X_train: training  dataset
    y_train: training dataset
    Returns
    -------
    A confusion matrix of given model
    '''
    fig, axes = plt.subplots(figsize=(13,8))
    #axes.set_title("Model Validation", fontsize=20)
    x_tick_marks = ['Predicted To stay', 'Predicted To leave']
    y_tick_marks = ['Stayed', 'Exited']
    
    plot_confusion_matrix(model, X_train, y_train, ax=axes, cmap='Blues', display_labels=y_tick_marks)
    plt.xticks([0,1], x_tick_marks)
    plt.title('Final model with RandomForest GridSearch')
    
    label_font = {'size':'20'}
    axes.set_xlabel('', fontdict=label_font)
    axes.set_ylabel('', fontdict=label_font)
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    confusion_matrix = plt.show()
    if save_path:
        plt.savefig(save_path, transparent=True)
    return confusion_matrix, fig
    