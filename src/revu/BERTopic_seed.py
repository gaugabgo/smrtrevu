from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pandas as pd

 # Load data
df = pd.read_csv("data/abstract_output/processed_EMRnbibparsed.csv")
documents = df['processed_text'].tolist()
# Define seed topics for EMR domain
seed_topic_list = [
    ["mean","median","mode", "variance","standarddeviation","range","interquartilerange","skewness","kurtosis","frequencydistribution","percentile","quartile","histogram","boxplot"],
    ["sample","confidence","hypothesis", "significance", "significant", "assumption", "parameter", "predict", "pvalue","ttest","ztest","anova", "standarderror",	"se", "degreesoffreedom", "df"],
    ["linearregression","multipleregression","regression","regressionmodel", "coefficient", "estimate", "adjustedrsquared","r","residual", "multicollinearity","homoscedasticity","autocorrelation","ordinaryleastsquare", "leastsquare","interactionterms"],
    ["counterfactual","potentialoutcome","averagetreatmenteffect","ate","instrumentalvariable","propensityscore","differenceindifference","regressiondiscontinuity","confounding","selectionbias","inference","causal", "bias"],
    ["traintest","crossvalidation", "validationsample", "classification","sensitivity", "specificity", "predict", "predictionmodel","overfit","underfit","fit", "accuracy","precision","recall",	"f1","roccurve","auc","decisiontree","randomforest","supportvectormachine","svm","gradientboosting", "segmentation", "segment", "algorithm"],
    ["cluster","dimensionality","principalcomponentanalysis", "pca","tsne","silhouettescore","elbowmethod","latentvariable","anomalydetection","anomaly", "associationrule","topicmodeling","nlp", "naturallanguageprocess", "algorithm"],
    ["prior","posterior","likelihood","bayes","bayesian","credibleinterval","markovchainmontecarlo","mcmc","gibbs","metropolishastings","conjugatepriors","hierarchicalmodel","bayesianinference","modelaveraging"],
    ["autocorrelation", "interval", "meaninterval", "seasonality", "partialautocorrelation","stationarity","arima","seasonaldecomposition","exponentialsmoothing","trend","seasonality","noise","forecasting","lag","unitroottest","holtwintersmethod"],
    ["manova","factoranalysis", "factor", "errorvariance", "canonicalcorrelation","multidimensionalscaling","principalcomponentanalysis","pca","clusteranalysis", "cluster","discriminantanalysis","covariancematrix","eigenvalue","eigenvector","latentvariable"],
    ["censoring","kaplanmeierestimator","hazardfunction","survivalfunction","coxproportionalhazardsmodel","logranktest","timetoevent","acceleratedfailuretime","schoenfeldresidual","mediansurvivaltime"],
    ["randomization","controlgroup","treatmentgroup","factorialdesign","block","blind","latinsquare","repeatedmeasures","interactioneffect","poweranalysis","power", "internalvalidity","externalvalidity","validity"],
    ["wilcoxonranksum","wilcoxon", "kruskalwallis","mannwhitneyu","chisquare","spearmanrankcorrelation","correlation", "spearman","signtest","bootstrap","permutationtest","kerneldensityestimation","rankbased"],
    ["simplerandomsampling","stratifiedsampling","clustersampling","systematicsampling","samplingframe","samplingbias","samplingdistribution","nonresponsebias","bias", "marginoferror","weighting"],
    ["validity","reliability","sensitivity", "specificity", "cronbachalpha","alpha","itemresponse","factorloading","factor", "likert","scaling","measurementerror","classicaltesttheory","standardscores","testretest","consistency","constructvalidity","criterionvalidity","contentvalidity"],
    ["bootstrap","jackknife","permutationtesting","crossvalidation","montecarlo","simulation","biasvariancetradeoff","bagging","ensemblemethods","randomizationtest","samplingwithreplacement","samplingwithoutreplacement"],
]


# Create guided BERTopic model
topic_model = BERTopic(
    seed_topic_list=seed_topic_list,
    representation_model=KeyBERTInspired(),
    verbose=True
)

# Fit the model
topics, probs = topic_model.fit_transform(documents)

#Print summary
def print_bertopic_summary(topic_model, documents, topics, seed_topic_list=None):
    """Print comprehensive BERTopic summary to terminal"""
    
    print("=" * 60)
    print("           BERTOPIC RESULTS SUMMARY")
    print("=" * 60)
    
    # Basic stats
    print(f"📊 Total documents: {len(documents)}")
    print(f"📊 Topics found: {len(set(topics))}")
    print(f"📊 Outliers: {topics.count(-1)} ({topics.count(-1)/len(topics)*100:.1f}%)")
    
    # Topic distribution
    topic_counts = pd.Series(topics).value_counts().sort_index()
    print(f"\n📈 Topic sizes (top 10):")
    for topic_id, count in topic_counts.head(10).items():
        if topic_id != -1:
            words = [w for w, s in topic_model.get_topic(topic_id)[:3]]
            print(f"   Topic {topic_id:2d}: {count:4d} docs - {', '.join(words)}")
    
    # Seed comparison if provided
    if seed_topic_list:
        print(f"\n🌱 Seed vs Generated Topics:")
        for i, seed in enumerate(seed_topic_list):
            if i in set(topics):
                generated = [w for w, s in topic_model.get_topic(i)[:3]]
                print(f"   Seed {i}: {seed}")
                print(f"   Gen  {i}: {generated}")
                print()
    
    print("=" * 60)

# Usage
print_bertopic_summary(topic_model, documents, topics, seed_topic_list)