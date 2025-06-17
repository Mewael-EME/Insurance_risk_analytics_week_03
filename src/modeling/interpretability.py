import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, preprocessor, feature_names):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig('shap_summary.png')
    shap.dependence_plot(feature_names[0], shap_values.values, X_train, show=False)
    plt.savefig('shap_dep.png')
