
library(readxl)
library(dplyr)
library(scales)
library(tidyr)
library(stats)
library(purrr)
library(ggplot2)
library(plotROC)
library(psych)
library(Hmisc)
library(caret)
library(epitools)
library(randomForest)
library(glmnet)
library(pROC)
library(precrec)
library(corrplot)
library(PRROC) 
library(ROCR)

setwd("~/Desktop/pituitary_ml/pituitary_data")
# Janitor work, tidy data for dataframe, predictors
Pituitarydb = read_excel("Pituitary_Outcomes_Study_Datasheets.xlsx", sheet = 3)
Pituitarydbcoledit = trimws(colnames(Pituitarydb))
Pituitarydbcoledit = make.names(colnames(Pituitarydb))
colnames(Pituitarydb) = Pituitarydbcoledit
Pituitarydb$TumorType[Pituitarydb$TumorType == "Cushing's Disease"] = "Cushing's" # Find and replace Tumor type names
Pituitarydb$TumorType[Pituitarydb$TumorType == "Cushing's disease"] = "Cushing's"
Pituitarydb$TumorType[Pituitarydb$TumorType == "Cushing's"] = "Cushing's" 
Pituitarydb$TumorType[Pituitarydb$TumorType == "cushings"] = "Cushing's" 
Pituitarydb$TumorType[Pituitarydb$TumorType == "acromegaly"] = "Acromegaly"
Pituitarydb$TumorType[Pituitarydb$TumorType == "nonsecreting"] = "Nonfunctioning" 
Pituitarydb$TumorType[Pituitarydb$TumorType == "Nonsecreting"] = "Nonfunctioning" 
Pituitarydb$TumorType[Pituitarydb$TumorType == "prolactinoma"] = "Prolactinoma" 
Pituitarydb = Pituitarydb[,-1]

# Add age column

date = Pituitarydb$DateofOperation
born = Pituitarydb$DateofBirth
age = function(date, born){
  as.numeric(as.Date(date) - as.Date(born)) / 365
}
agevector = mapply(age, date, born)
Pituitarydb = data.frame(Pituitarydb, as.data.frame(agevector))

# defining the relevant tumortype vectors for odds ratios
cushingvector = as.numeric(Pituitarydb$TumorType == "Cushing's")
acromegalyvector = as.numeric(Pituitarydb$TumorType == "Acromegaly")
nonfunctioningvector = as.numeric(Pituitarydb$TumorType == "Nonfunctioning")
prolactinomavector = as.numeric(Pituitarydb$TumorType == "Prolactinoma")
tshomevector = as.numeric(Pituitarydb$TumorType == "TSHoma")
sum(c(cushingvector, acromegalyvector, nonfunctioningvector, prolactinomavector, tshomevector)) == 400

#################Constucting features, target variables 1 (inpatient morbidity) and 2 (outcome-day morbidity)
feature_select = select(Pituitarydb,
                        # Demographics
                        TumorType, 
                        Macroadenoma,
                        agevector,
                        Gender,
                        Race,
                        # comorbidities
                        BMI,
                        HoMI, 
                        HoCHF, 
                        HoStroke, 
                        Immunesuppression,
                        DM,
                        Hopulmonarydisease,
                        RenalDisease,
                        LiverDisease,
                        # medication risk factors
                        Bloodthinners,
                        # cranial nerve deficits
                        Hopriorpituitarysurgery,
                        Horadiationtoskullbase,
                        PreopVisualacuity,
                        Preopvisualfieldcut,
                        # postoperative risk factors
                        CranialNerveInjury,
                        PostopVisualfieldchange,
                        PostopVisualacuitychange,
                        # Endocrine issues
                        PostopNalowest,
                        PostopNahighest, 
                        DiabetesInsipidus,
                        DesmopressinRequired)


### Define Outcome vector
outcome_day_vector = transmute(Pituitarydb, outcomes = Day30mortality + Day30readmission  + Day30EDvisits +
                                Stroke + DVTorPE + SevereArrhythmia + RespiratoryFailure + CSFLeak + TensionPneumocephalus + IntracranialInfection)
### Define the Greater than expected length of stay vector
lengthofstay = function(tumor, los) {
  if (tumor == "Cushing's" && los > 5) {
    return(1)
  } else if (los > 3) {
    return(1)
  } else {
    return(0)
  }
}
Non_Cushing_df = Pituitarydb[Pituitarydb$TumorType != "Cushing's",]
Non_Cushing_extended_los = Non_Cushing_df[Non_Cushing_df$LOS.days. > 2,]
Cushings_df = Pituitarydb[Pituitarydb$TumorType == "Cushing's",]
Cushing_extended_los = Cushings_df[Cushings_df$LOS.days. > 4,]

LOS_binary = as.numeric(mapply(lengthofstay, Pituitarydb$TumorType, Pituitarydb$LOS.days.))
temp_df = data.frame(outcome_day_vector, LOS_binary)
summed_outcome = transmute(temp_df, outcome = outcomes + LOS_binary)
outcome_binary = ifelse(summed_outcome > 0, 1, 0)

# Concatenate features and outcomes to final dataframe
outcome_df = data.frame(feature_select, outcome_binary)

# Outcome tallied by category
outcome_tally = select(Pituitarydb, DVTorPE, MI, SevereArrhythmia, RespiratoryFailure, Stroke,
                       CSFLeak, TensionPneumocephalus, IntracranialInfection,
                       Day30readmission,  Day30EDvisits, Day30mortality)
outcome_tally = data.frame(outcome_tally, LOS_binary)
colnames(outcome_tally) = c("DVTorPE", "MI", "Arrhythmia", "Resp. failure", "Stroke", "CSF leak",
                            "Sympt. pneumocephalus", "Meningitis", "Readmission", "ED readmission", "Death", "Ext. LOS")

outcome_tally_gather = gather(outcome_tally)
ggplot(outcome_tally_gather, aes(x = reorder(key, value), y=value)) + 
  geom_histogram(stat='identity') + 
  xlab("") + 
  ylab("") +
  # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(text = element_text(size=14)) +
  scale_y_continuous(breaks=seq(0,70,5)) +
  coord_flip()
  
ggplot(outcome_tally_gather, aes(x = reorder(key, value), y=value)) + 
  geom_histogram(stat='identity') + 
  xlab("") + 
  ylab("") +
  # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(text = element_text(size=14)) +
  scale_y_continuous(breaks=seq(0,70,5)) 

# Histogram of number of outcomes 
summed_outcome_trim = filter(summed_outcome, outcome != 7)
ggplot(summed_outcome_trim, aes(outcome)) +
  geom_histogram(binwidth = 1, fill = "grey50", colour = "white") +
  scale_y_continuous(breaks=seq(0,275,25)) +
  scale_x_continuous(breaks=seq(0,6,1))

#### Descriptive Data Visualization
# Sex by tumor type
ggplot(outcome_df, aes(x = factor(1), fill = TumorType)) +
  geom_bar(width = 1) +
  facet_grid(.~Gender, scales = "free") 

# Tumor size by tumor type 
ggplot(outcome_df, aes(x = factor(1), fill = TumorType)) +
  geom_bar()  +     
  facet_grid(Macroadenoma~., scales = "free") +
  coord_flip() 

# Age by tumor type figure
posn.j = position_jitter(0.2)
ggplot(outcome_df, aes(x = TumorType, y = agevector, col = TumorType)) +
  geom_jitter(position = posn.j) +
  xlab("") +
  ylab("Age") +
  labs(col = "Tumor Type") + 
  scale_y_continuous(breaks = c(seq(10,90, 10))) +
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.8) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1),
               geom = "errorbar", width = 0.3, col = "black", alpha = 0.8)

# BMI by tumor type figure
posn.j = position_jitter(0.2)
ggplot(outcome_df, aes(x = TumorType, y = BMI, col = TumorType)) +
  geom_jitter(position = posn.j) +
  xlab("") +
  ylab("BMI") +
  scale_y_continuous(breaks = c(seq(10,70, 5))) +
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.8) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1),
               geom = "errorbar", width = 0.3, col = "black", alpha = 0.8)

##########################Machine Learning 
# Define helper functions
change_levels = function(x){
  if (x == 0){
    return("No")
  } else if (x == 1){
    return("Yes")
  }
}
race_function = function(x){
  if (x == 'W'){
    return('white')
  } else if (x == 'B'){
    return('black')
  } else {
    return('other')
  }
}

# Need to assign appropriate type to all variables
col_name_list = colnames(outcome_df)
outcome_df$Gender = as.factor(outcome_df$Gender)
outcome_df$Race = as.factor(map_chr(outcome_df$Race, race_function))
outcome_df$TumorType = as.factor(outcome_df$TumorType)

# Set arithmetic to identify two level factor variables
two_level_vars = setdiff(colnames(outcome_df), c("agevector", "Gender", "Race", "TumorType", "BMI", "PostopNalowest", "PostopNahighest"))
# Change levels
for (cols in two_level_vars){
  print(cols)
  outcome_df[[cols]] = as.factor(map_chr(outcome_df[[cols]], change_levels))
}

# Continues variable, rescaled
range_zero_one = function(x){
  y = (x-min(x))/(max(x)-min(x))
  return(y)
}
outcome_df$agevector = range_zero_one(outcome_df$agevector)
outcome_df$BMI = range_zero_one(outcome_df$BMI)
outcome_df$PostopNalowest = range_zero_one(outcome_df$PostopNalowest)
outcome_df$PostopNahighest = range_zero_one(outcome_df$PostopNahighest)

# Define the the outcome as binary                             
outcome_df$outcome = as.factor(outcome_df$outcome)
colnames(outcome_df) = col_name_list

########## Multivariate analysis of postoperative conditions
# DIABETES INSIPIDUS, Multivariate analysis
DI_df = cbind(select(Pituitarydb, agevector, Macroadenoma, DiabetesInsipidus), cushingvector)
DI_df$Macroadenoma = ifelse(DI_df$Macroadenoma == 1, 0, 1) # flips to microadenomas for easier analysis
DI_df$agevector = ifelse(DI_df$agevector > 40, 1, 0)
log_reg_DI = glm(DiabetesInsipidus~., data = DI_df, family = 'binomial')
summary(log_reg_DI)
exp(cbind(OR = coef(log_reg_DI), confint(log_reg_DI)))

DI_df = cbind(select(Pituitarydb, agevector, Macroadenoma, DiabetesInsipidus), cushingvector)
ggplot(Pituitarydb, aes(x = as.factor(DiabetesInsipidus), y = agevector, col = as.factor(DiabetesInsipidus))) + 
  geom_jitter(position = posn.j, alpha = 0.3) +
  xlab("Diabetes Insipidus") +
  ylab("Age") +
  scale_y_continuous(breaks = c(seq(10, 100, 5))) +
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.8) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1),
               geom = "errorbar", width = 0.3, col = "black", alpha = 0.8)
# Density plot
ggplot(Pituitarydb, aes(x = agevector, fill = as.factor(DiabetesInsipidus))) + 
  geom_density()

temp_df = Pituitarydb[Pituitarydb$PostopNahighest > 130,]
ggplot(temp_df, aes(x = agevector, y = PostopNahighest, col = as.factor(DiabetesInsipidus), size = as.factor(DesmopressinRequired))) +
  geom_jitter(alpha = 0.75) # Be sure to include the trimmed mean for this figure

mean(Pituitarydb[Pituitarydb$DiabetesInsipidus == 1,]$agevector, trim = 0.05)
mean(Pituitarydb[Pituitarydb$DiabetesInsipidus != 1,]$agevector, trim = 0.05)

# HYPONATREMIA, multivariate analysis
hyponat_df = select(Pituitarydb, Hyponatremia, agevector, Obesity, TumorType)
hyponat_df$agevector = ifelse(hyponat_df$agevector > 40, 1, 0)
log_reg_hyponat = glm(Hyponatremia~., data = hyponat_df, family = 'binomial')
summary(log_reg_hyponat)
exp(cbind(OR = coef(log_reg_hyponat), confint(log_reg_hyponat)))

ggplot(Pituitarydb, aes(x = agevector, y = PostopNalowest, col = as.factor(Hyponatremia))) +
  geom_jitter(alpha = 0.75)

pit_bmi_trim = Pituitarydb[Pituitarydb$BMI < 50,]
ggplot(pit_bmi_trim, aes(x = BMI, fill = as.factor(Hyponatremia))) + 
  geom_density(alpha = 0.5)
mean(Pituitarydb[Pituitarydb$Obesity == 1,]$PostopNalowest, trim = 0.05)
mean(Pituitarydb[Pituitarydb$Obesity != 1,]$PostopNalowest, trim = 0.05)

# DVT/PE, multivariate analysis
dvt_df = cbind(select(Pituitarydb, DVTorPE, HoCHF, Bloodthinners), cushingvector)
# DI_df$agevector = ifelse(DI_df$agevector > 40, 1, 0)
log_reg_DI = glm(DVTorPE~., data = dvt_df, family = 'binomial')
summary(log_reg_DI)
exp(cbind(OR = coef(log_reg_DI), confint(log_reg_DI)))

######### Biulding machine learning algorithms
## Train/Test split correcting for class imbalance
# Developing balanced testing set 
outcome_pit_df = filter(outcome_df, outcome == 'Yes')
nooutcome_pit_df = filter(outcome_df, outcome == 'No')
set.seed(12345) 
test_outcome_df = sample_n(outcome_pit_df, size = 25, replace = FALSE)
test_nooutcome_df = sample_n(nooutcome_pit_df, size = 75, replace = FALSE)
test_outcome_one = rbind(test_outcome_df, test_nooutcome_df)
# Training set that excludes all examples in testing set
train_outcome_one = anti_join(outcome_df, test_outcome_one)

##### ML Algorithm training and testing
# 1) Naive Bayes
# 2) SVM
# 3) RandomForest 
# 4) GLMnet
#### Train and cross validate models >>>> evaluate using testing set
# Naive Bayes
myControl = trainControl(method = "cv", number = 10, classProbs = TRUE, verboseIter = TRUE) 
pit_nb = train(outcome~., data = train_outcome_one, trControl = myControl, method="nb")
pit_nb_pred = predict(pit_nb, test_outcome_one)
confusionMatrix(pit_nb_pred, test_outcome_one$outcome, positive = "Yes")

# Support vector machine with linear kernel 
set.seed(12345) 
myControl = trainControl(method = "cv", number = 10, classProbs = TRUE, verboseIter = TRUE) 
grid = data.frame(C = seq(1,10))
pit_svm = train(outcome~., data = train_outcome_one, trControl = myControl, tuneGrid = grid, method = "svmLinear")
pit_svm_pred = predict(pit_svm, test_outcome_one)
confusionMatrix(pit_svm_pred, test_outcome_one$outcome, positive = "Yes")

# Random Forest
set.seed(12345)
grid = data.frame(mtry = seq(1,length(train_outcome_one)))
myControl = trainControl(method = "cv", number = 10, classProbs = TRUE, verboseIter = TRUE)
pit_rf = train(outcome~., data = train_outcome_one, tuneGrid = grid, trControl = myControl, method = "rf", ntree = 500)
pit_rf_pred = predict(pit_rf, test_outcome_one)
confusionMatrix(pit_rf_pred, test_outcome_one$outcome, positive = "Yes")

# ElasticNet
set.seed(12345)
myControl = trainControl(method = "cv", number = 10, classProbs = TRUE, verboseIter = TRUE) # accuracy
pit_glm = train(outcome~., data = train_outcome_one, tuneGrid = data.frame(alpha = c(0, 0.05, 0.1, 0.9, 0.95, 1), lambda = c(0.0, 0.005, 0.01, 0.1, 1, 10)),  trControl = myControl, method = 'glmnet')
pit_glm_pred = predict(pit_glm, test_outcome_one)
confusionMatrix(pit_glm_pred, test_outcome_one$outcome, positive = "Yes")

#### Evaluate accuracy of models ####
model_list = list(pit_nb, pit_svm, pit_rf, pit_glm)
accuracy_function = function(model_list, dataframe){
  accs = c()
  for (model in model_list) {
    preds = predict(model, test_outcome_one)
    acc = confusionMatrix(preds, dataframe$outcome)$overall[1]
    accs = c(accs, acc) 
    } 
  return(accs)
}
test_accuracy_values = accuracy_function(model_list, test_outcome_one)

#######Mcnemar tests for statistically significantly different models
prediction_df = data.frame(pit_nb_pred, pit_svm_pred, pit_rf_pred, pit_glm_pred)
pvalue_rescaling = function(x) {
  if (x > 0.05 | is.na(x)) {
    return(0)
  } else {
    return(log(1/x))
  }
}

pairwise_mcnemar = function(prediction_df){
  paired_mcnemar = c()
  for (col1 in prediction_df){
    for (col2 in prediction_df){
      pval = mcnemar.test(col1, col2)$p.value
      paired_mcnemar = c(paired_mcnemar, pval)
    }
  }
  paired_rescaled = map_dbl(paired_mcnemar, pvalue_rescaling)
  paired_mcnemar <- matrix(paired_rescaled, nrow = 4, byrow = TRUE)
  return(paired_mcnemar)
}
test = pairwise_mcnemar(prediction_df)
image(test, col=viridis(256))

###### Correlation of predictions among models
one_zero = function(x){
  if (x == "Yes"){
    return(1)
  } else{
    return(0)
  }
}
prediction_df$pit_nb_pred = as.numeric(map(prediction_df$pit_nb_pred, one_zero))
prediction_df$pit_svm_pred = as.numeric(map(prediction_df$pit_svm_pred, one_zero))
prediction_df$pit_rf_pred = as.numeric(map(prediction_df$pit_rf_pred, one_zero))
prediction_df$pit_glm_pred = as.numeric(map(prediction_df$pit_glm_pred, one_zero))
corr_matrix = cor(as.matrix(prediction_df))
corrplot(corr_matrix, method="pie", type="upper", order="hclust")

####### Generate prediction probablities and ROC curves ########
# ROC for training data
roc_function = function(model_list, dataframe) {
  rocs = c()
  for (model in model_list) {
    predictions = predict(model, dataframe, type = "prob")[,2]
    roc_val = roc(dataframe[, ncol(dataframe)], predictions)
    rocs = c(rocs, roc_val$auc) 
    } 
  return(rocs)
}

roc_function_aucs = function(model_list, dataframe) {
  rocs = c()
  for (model in model_list) {
    predictions = predict(model, dataframe, type = "prob")[,2]
    roc_val = roc(dataframe[, ncol(dataframe)], predictions)
    rocs = c(rocs, roc_val) 
    } 
  return(rocs)
}

# Train ROC values
train_rocs_values = roc_function(model_list, train_outcome_one)
# Train prediction vectors
pit_nb_pred_probs = predict(pit_nb, train_outcome_one, type = "prob")[,2]
pit_enet_pred_probs = predict(pit_glm, train_outcome_one, type = "prob")[,2]
pit_rf_pred_probs = predict(pit_rf, train_outcome_one, type = "prob")[,2]
pit_svm_pred_probs = predict(pit_svm, train_outcome_one, type = "prob")[,2]
# ROC plots for training set
ROC_df = data_frame(pit_nb_pred_probs, pit_enet_pred_probs, pit_rf_pred_probs, pit_svm_pred_probs, train_outcome_one$outcome)
ROC_df = rename(ROC_df, outcome = `train_outcome_one$outcome`)
ROC_gathered = gather(ROC_df, key = "model", value = "probability", -outcome)
ggplot(ROC_gathered, aes(d = outcome, m = probability, color = model))+
  geom_roc(n.cuts = 0) +
  style_roc()

###### ROC for testing data
# Test ROC values
test_rocs_values = roc_function(model_list, test_outcome_one)
# Test ROC values
pit_nb_pred_probs = predict(pit_nb, test_outcome_one, type = "prob")[,2]
pit_enet_pred_probs = predict(pit_glm, test_outcome_one, type = "prob")[,2]
pit_rf_pred_probs = predict(pit_rf, test_outcome_one, type = "prob")[,2]
pit_svm_pred_probs = predict(pit_svm, test_outcome_one, type = "prob")[,2]
# ROC plots for test set
ROC_df = data_frame(pit_nb_pred_probs, pit_svm_pred_probs, pit_rf_pred_probs, pit_enet_pred_probs, test_outcome_one$outcome)
ROC_df = rename(ROC_df, outcome = `test_outcome_one$outcome`)
ROC_gathered = gather(ROC_df, key = "model", value = "probability", -outcome)
ggplot(ROC_gathered, aes(d = outcome, m = probability, color = model)) +
  geom_roc(n.cuts = 0) +
  style_roc()

########### Precision recall curves/AUC
prauc_df = data_frame(pit_nb_pred_probs, pit_svm_pred_probs, pit_rf_pred_probs, pit_enet_pred_probs)
prauc_function = function(prediction_probs_df) {
  praucs = c()
  for (preds in prediction_probs_df) {
    scores <- data.frame(preds, test_outcome_one$outcome)
    pr <- pr.curve(scores.class0=scores[scores$test_outcome_one.outcome=="Yes",]$preds,
             scores.class1=scores[scores$test_outcome_one.outcome=="No",]$preds, curve = FALSE)
    praucs = c(praucs, pr$auc.integral)
  }
  return(praucs)
}
test_prauc_value = prauc_function(prauc_df)

# Precision-recall curves
eval = evalmod(scores = pit_enet_pred_probs, labels = test_outcome_one$outcome)
autoplot(eval1)
# Use a list with multiple score vectors and a list with a single label vector
predslist = list(pit_nb_pred_probs, pit_enet_pred_probs, pit_rf_pred_probs, pit_svm_pred_probs)
# Explicitly specify model names
msmdat2 = mmdata(predslist, test_outcome_one$outcome, modnames = c("NaiveBayes", "ElasticNet", "RandomForest", "SupportVectorMachines"))
# Use a sample dataset created by the create_sim_samples function
mscurves = evalmod(msmdat2)
autoplot(mscurves)
# autoplot(mscurves, "PRC")

####### Plotting probabilities and outcomes labels for random forest
enet_df = data_frame(test_outcome_one$outcome, pit_enet_pred_probs, 1:100)
ggplot(enet_df, aes(y = pit_enet_pred_probs, x = `1:100`, color = test_outcome_one$outcome)) + 
  geom_point() + 
  scale_y_continuous(limits = c(0, 1))
mean(enet_df[enet_df$`test_outcome_one$outcome` == "No",]$pit_enet_pred_probs, trim = 0.05)
sd(enet_df[enet_df$`test_outcome_one$outcome` == "No",]$pit_enet_pred_probs)

enet_df$pit_enet_pred_probs = enet_df$pit_enet_pred_probs - 0.5
enet_df <- enet_df[order(pit_enet_pred_probs),]

ggplot(enet_df, aes(x=seq_along(pit_enet_pred_probs), y = pit_enet_pred_probs)) + 
  geom_bar(stat = 'identity', aes(fill = `test_outcome_one$outcome`), position = 'dodge', col = 'transparent') + 
  scale_fill_discrete(guide = 'none') + 
  labs(x = '', y = '')

test = data.frame(varImp(pit_rf)$importance)
df = data.frame(row.names(test), test$Overall)
df = df[df$test.Overall > 13.69,]
df = df[order(df$test.Overall),]

ggplot(df, aes(x = row.names.test., y = test.Overall)) +
  geom_bar(stat = 'identity')

# save(pit_glm, file = "Pit_elasticnet.rda")
# load("Pit_elasticnet.rda")

outcome_df = group_by(outcome_df, TumorType)
summarise(outcome_df, mean = mean(agevector))
summarise(outcome_df, sd = sd(agevector))
foo = outcome_df[outcome_df$TumorType != "Nonfunctioning", ]
bar = outcome_df[outcome_df$TumorType == "Nonfunctioning", ]
mean(foo$agevector)

Pituitarydb = group_by(Pituitarydb, TumorType)
summarise(Pituitarydb, mean = mean(BMI))
summarise(Pituitarydb, sd = sd(BMI))
foo = outcome_df[outcome_df$TumorType != "Nonfunctioning", ]
bar = outcome_df[outcome_df$TumorType == "Nonfunctioning", ]





