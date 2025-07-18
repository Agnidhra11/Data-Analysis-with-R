getwd()
setwd("C:/Users/Agnidhra Banerjee/Desktop")

df_housing_train=read.csv("housing_train.csv", stringsAsFactors = F)

df_housing_test=read.csv("housing_test.csv", stringsAsFactors = F)

View(df_housing_train)
sort(table(df_housing_train$Suburb),decreasing = T)
table(df_housing_train$Type)

tapply(mtcars$mpg, mtcars[,c("am","vs")],mean)
mtcars$am[mtcars$am==1]

x1=mean(df_housing_train$Price[df_housing_train$Type=="h"])
x2=mean(df_housing_train$Price[df_housing_train$Type=="t"])
x1-x2
sort(tapply(df_housing_train$Price, df_housing_train[,7],sum),decreasing = T)[1:3]
sort(tapply(df_housing_train$Price, df_housing_train[,16],var),decreasing = T)[1:3]
sort(table(df_housing_train$SellerG), decreasing = T)
var(df_housing_train$Price)
table(df_housing_train$YearBuilt)

lapply(df_housing_train,function(x) sum(is.na(x)))
sum(is.na(df_housing_train$YearBuilt))

p=ggplot(df_housing_train,aes(x=Distance))
p+geom_bar()
p+geom_line()

shapiro.test(t1$Distance)

set.seed(2)
s=sample(1:nrow(df_housing_train),0.65*nrow(df_housing_train))
t1=df_housing_train[s,]
View(t1)
unique(df_housing_train$Postcode)
table(unique(df_housing_train$Postcode))

summary(df_housing_train)
glimpse(df_housing_train)

table(df_housing_train$CouncilArea)
sort(table(df_housing_train$Suburb),decreasing = T)
unique(df_housing_train$Suburb)
sort(table(df_housing_train$Postcode),decreasing = T)

sort(tapply(df_housing_train$Price, df_housing_train[,1],mean),decreasing = T)
sort(table(df_housing_train$SellerG),decreasing = T)
sort(table(df_housing_train$Method),decreasing = T)
sort(table(df_housing_train$YearBuilt),decreasing = T)

df_housing_train$YearBuilt=as.character(df_housing_train$YearBuilt)
sort(table(df_housing_train$Postcode),decreasing = T)

df_housing_train$Postcode=as.character(df_housing_train$Postcode)

df_housing_train$Price=as.factor(as.numeric(df_housing_train$Price))

glimpse(df_housing_train)

dp_pipe=recipe(Price ~ .,data=df_housing_train) %>% 
  update_role(Address,Postcode,new_role = "drop_vars") %>% 
  update_role(Suburb,Type,Method,SellerG,YearBuilt,CouncilArea,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.01,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)
View(train)

lapply(train,function(x) sum(is.na(x)))

test=bake(dp_pipe,new_data=df_housing_test)
View(test)

set.seed(2)
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,]
t2=train[-s,]

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

folds = vfold_cv(train, v = 5)

tree_grid = grid_regular(cost_complexity(), tree_depth(),
                         min_n(), levels = 3)

my_res=tune_grid(
  tree_model,
  Price~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
  
)






fit=lm(Price~. -Type_X__other__ -Suburb_X__other__ 
       -YearBuilt_X__missing__ ,data=t1)

sort(vif(fit),decreasing = T)[1:3]

fit=stats::step(fit)

summary(fit)

formula(fit)

fit=lm(Price ~ Rooms + Distance + Bedroom2 + Bathroom + Car + 
         BuildingArea + Suburb_Balwyn + Suburb_Balwyn.North + Suburb_Bentleigh + 
         Suburb_Brighton + Suburb_Brighton.East + Suburb_Camberwell + 
         Suburb_Carnegie + Suburb_Doncaster + Suburb_Elwood + Suburb_Essendon + 
         Suburb_Glen.Iris + Suburb_Glenroy + Suburb_Hampton + Suburb_Hawthorn + 
         Suburb_Keilor.East + Suburb_Kew + Suburb_Malvern.East +  
         Suburb_Newport + Suburb_Port.Melbourne + Suburb_Prahran + 
         Suburb_Preston + Suburb_Reservoir + Suburb_Richmond + Suburb_Thornbury + 
         Suburb_Toorak + Type_t + Type_u + Method_S + Method_SP + 
         Method_X__other__ +  SellerG_Buxton + 
         SellerG_Greg + SellerG_Jellis + SellerG_Kay + SellerG_Marshall + 
          SellerG_Miles + SellerG_RT + SellerG_Sweeney + 
          YearBuilt_X1900 + YearBuilt_X1910 + YearBuilt_X1950 + 
         YearBuilt_X1960 + YearBuilt_X1970 +  YearBuilt_X2005 + 
          CouncilArea_Banyule + CouncilArea_Bayside + 
         CouncilArea_Boroondara + CouncilArea_Brimbank + CouncilArea_Darebin + 
         CouncilArea_Hobsons.Bay + CouncilArea_Manningham + CouncilArea_Maribyrnong + 
         CouncilArea_Melbourne + CouncilArea_Moonee.Valley + CouncilArea_Moreland + 
          CouncilArea_Whitehorse + CouncilArea_Yarra + 
         CouncilArea_X__other__, data=t1)

summary(fit)

t2.pred=predict(fit,newdata=t2)





fit=lm(Price ~ Rooms + Distance + Bedroom2 + Bathroom + Car + Landsize + 
         BuildingArea + Type_t + Type_u + Method_S + Method_SP +  
         SellerG_Buxton + SellerG_Fletchers + SellerG_Greg + SellerG_Jellis + 
         SellerG_Marshall + SellerG_Miles + SellerG_RT + SellerG_X__other__ + 
         Postcode_X3012 + Postcode_X3020 + Postcode_X3032 + Postcode_X3040 + 
         Postcode_X3046  + Postcode_X3058 + Postcode_X3072 + 
         Postcode_X3073 + Postcode_X3121  + Postcode_X3163 + 
         Postcode_X3181 + Postcode_X3186 + Postcode_X3204 + YearBuilt_X1900 + 
         YearBuilt_X1910 + YearBuilt_X1950 + YearBuilt_X1960 + YearBuilt_X1970 + 
           CouncilArea_Banyule + 
         CouncilArea_Bayside + CouncilArea_Boroondara + CouncilArea_Brimbank + 
         CouncilArea_Darebin + CouncilArea_Glen.Eira + CouncilArea_Hobsons.Bay + 
         CouncilArea_Manningham + CouncilArea_Maribyrnong + CouncilArea_Melbourne + 
         CouncilArea_Moonee.Valley + CouncilArea_Moreland + CouncilArea_Stonnington + 
         CouncilArea_Yarra + CouncilArea_X__other__,data=t1)


t2.pred=predict(fit,newdata=t2)

errors=t2$Price-t2.pred

rmse=errors**2 %>% mean() %>% sqrt()
mae=mean(abs(errors))
212467/rmse

fit.final=lm(Price ~.-Type_X__other__ -Suburb_X__other__
             -YearBuilt_X__missing__,data=train)

sort(vif(fit.final),decreasing = T)[1:3]

fit.final=stats::step(fit.final)

summary(fit.final)

formula(fit.final)

fit.final=lm(Price ~ Rooms + Distance + Bedroom2 + Bathroom + Car + Landsize + 
               BuildingArea + Suburb_Balwyn + Suburb_Balwyn.North + Suburb_Bentleigh + 
                Suburb_Brighton + Suburb_Brighton.East + 
               Suburb_Camberwell + Suburb_Carnegie +  Suburb_Doncaster + 
               Suburb_Elwood + Suburb_Essendon + Suburb_Glen.Iris + Suburb_Glenroy + 
               Suburb_Hampton + Suburb_Hawthorn + Suburb_Keilor.East + Suburb_Kew + 
               Suburb_Malvern.East + Suburb_Newport + Suburb_Port.Melbourne + 
               Suburb_Prahran + Suburb_Preston + Suburb_Reservoir + Suburb_Richmond + 
                Suburb_Thornbury + Suburb_Toorak + Type_t + 
               Type_u + Method_S + Method_SP +   
               SellerG_Buxton +  SellerG_Greg + SellerG_Jellis + 
               SellerG_Kay + SellerG_Marshall +  SellerG_Miles + 
               SellerG_RT +   YearBuilt_X1900 + 
               YearBuilt_X1910 +   YearBuilt_X1950 + 
               YearBuilt_X1960 + YearBuilt_X1970 +  CouncilArea_Banyule + 
               CouncilArea_Bayside + CouncilArea_Boroondara + CouncilArea_Brimbank + 
               CouncilArea_Darebin + CouncilArea_Hobsons.Bay + CouncilArea_Manningham + 
               CouncilArea_Maribyrnong + CouncilArea_Melbourne + CouncilArea_Moonee.Valley + 
               CouncilArea_Moreland + CouncilArea_Whitehorse + CouncilArea_Yarra + 
               CouncilArea_X__other__ , data=train)


summary(fit.final)




summary(fit.final)

test.pred=predict(fit.final,newdata=test)
train.pred=predict(fit.final, newdata=train)



errors=train$Price-train.pred

rmse=errors**2 %>% mean() %>% sqrt()
mae=mean(abs(errors))
212467/rmse_xgb

write.csv(test.pred,"Agnidhra_Banerjee_P1_part2.csv",row.names = F)
