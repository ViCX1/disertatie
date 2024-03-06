# Calculate and print accuracy and precision
    if 'label' in df_test.columns:
        y_test = df_test['label']
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print(f'Accuracy: {accuracy*100:.2f}%')
        print(f'Precision: {precision*100:.2f}%')

        # Compute ROC curve and ROC area for each class
        y_pred_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()