## drop columns with too many missing values
round(colMeans(is.na(df_housing_train)),2)



##to delete all rows contain NA value
df_housing_train=df_housing_train %>% na.omit()

setwd("C:/Users/Agnidhra Banerjee/Desktop")

df_housing_train=read.csv("housing_train.csv", stringsAsFactors = F)

df_housing_test=read.csv("housing_test.csv", stringsAsFactors = F)

df_housing_train$YearBuilt=as.character(df_housing_train$YearBuilt)

df_housing_train$Postcode=as.character(df_housing_train$Postcode)

df_housing_train$Price=as.factor(as.numeric(df_housing_train$Price))


dp_pipe=recipe(Price ~ .,data=df_housing_train) %>% 
  update_role(Suburb,Address,new_role = "drop_vars") %>% 
  update_role(Type,Method,SellerG,Postcode,YearBuilt,CouncilArea,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.015,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)

test=bake(dp_pipe,new_data=df_housing_test)


##xgboost

xgb_spec = boost_tree(
  trees = 1000,  ## 1000 is a better value
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune(),
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")


xgb_grid = grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(),train),
  learn_rate(),
  size = 30  ## 30-50 is a good number
)


xgb_wf = workflow() %>% 
  add_formula(Price ~ .) %>% 
  add_model(xgb_spec)

set.seed(123)
power_folds = vfold_cv(train,v=10) ## v=10 is ideal

set.seed(234)
xgb_res = tune_grid(
  xgb_wf,
  resamples = power_folds,
  grid = xgb_grid,
  control = control_grid(verbose = TRUE)
)

install.packages("xgboost")
library(xgboost)
collect_metrics(xgb_res)

xgb_res %>% 
  collect_metrics() %>% 
  filter(metrics="rmse") %>% 
  select(mean,mtry:sample_size) %>% 
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
               ) %>% 
  ggplot(aes(value,mean,color = parameter)) +
  geom_point(alpha= 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "rmse")

show_best(xgb_res,"rmse")

best_rmse = select_best(xgb_res,"rmse")


final_xgb = finalize_workflow(
  xgb_wf,
  best_rmse
)

library(vip)

final_xgb_fit=final_xgb %>% 
  fit(data = train) %>% 
  pull_workflow_fit()

final_xgb_fit %>% vip(geom = "point")

test_forecast_xgb = predict(final_xgb_fit,new_data = test )
View(test)
x=train[,-8]
View(x)

train_forecast_xgb = predict(final_xgb_fit, new_data = x)
View(train_forecast_xgb)

rmse_xgb = mean((train$Price - train_forecast_xgb$.pred)^2) %>% sqrt()

write.csv(test_forecast_xgb,"Agnidhra_Banerjee_P1_part2.csv",row.names = F)
