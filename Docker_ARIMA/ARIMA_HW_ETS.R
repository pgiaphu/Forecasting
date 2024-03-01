library(bigrquery)
library(magrittr)
library("dplyr")
library("reshape2")
library("forecast")
library(lubridate)
library(zoo)
#
#Sys.setenv("GOOGLE_APPLICATION_CREDENTIALS" = "D:/OneDrive - pnj.com.vn/SA/service-account-uat.json")
#bq_auth(path = Sys.getenv("D:/OneDrive - pnj.com.vn/SA/service-account-uat.json"))

hor = 13

#QUERY FROM BIGQUERY
projectid <- "pnj-sc-aa-enhance"
sql <- "  SELECT
            YEAR_MONTH,
            REGION,
            PFSAP,
            CAST(ADJ_QTY AS FLOAT64) AS QNT_ADJ
          FROM
            `pnj-sc-aa-enhance.INPUT_FORECAST.W_INPUT_FC_DATABYMONTH_AGG`
          WHERE
            PFSAP NOT IN ('None')  
      "
#ACTUAL SALE TO DATAFRAME
df <- bq_project_query(projectid, sql) %>% bq_table_download()
df <- subset(df, YEAR_MONTH != format(with_tz(Sys.time(), tzone = "Asia/Singapore"), "%Y-%m")   )


data1 <- dcast(df, YEAR_MONTH ~ REGION + PFSAP , value.var = "QNT_ADJ", fun.aggregate = sum)
data <- data1[]
his_data <- as.data.frame(data1[-1])
pf_list <- colnames(data)[-1]
#----Set condition to take short time line Reg~PFSAP list----#
mva_list <- list()
for (i in 1:length(pf_list)){
  fil_list<-data1[,pf_list[i]]
  c_point <- length(data1$YEAR_MONTH) - sum(is.na(fil_list))
  c_point_df <- as.data.frame(c_point)
  colnames(c_point_df)[1]<-paste(pf_list[i])
  mva_list[[i]]<-c_point_df # assign the df value to the loop
}
mva<-as.data.frame(mva_list)
#----Pick up Reg~PFSAP where is the time line < 12 ----#
shr_mva1 <- mva[which(mva<=12)]
shr_mva2 <- shr_mva1[which(1<shr_mva1)]
shr_mva_df <- as.data.frame(shr_mva2)
shr_mva_list <- colnames(shr_mva_df)
nor_pf_list<- pf_list[!pf_list %in% shr_mva_list]
#-------------------Forecast with models----------------------------#
data_clean <- list()
for (i in 1:length(nor_pf_list)) {
  test <- data1[ ,nor_pf_list[i]]
  test_clean <- test
  test_clean[is.na(test_clean)] <- 1
  test[is.na(test)] <- quantile(test_clean, probs = .5, na.rm = T)
  qnt <- quantile(test, probs = c(.25, .75), na.rm = T)
  H <- 1.5 *(qnt[2] - qnt[1])
  test[test < (qnt[1] - H)] <- max(qnt[1] - H,0) 
  test[test > (qnt[2] + H)] <- max(qnt[2] + H,0)
  test <- as.data.frame(test)
  colnames(test)[1] <- paste(nor_pf_list[i])
  data_clean[[i]] <- test 
}
data_clean <- as.data.frame(data_clean)
list <- colnames(data_clean)
#--------------------- HOLT WINTER ADDITIVES ---------------------#
fc_hw1 <- data.frame(matrix(nrow = hor, ncol = 0))
for (i in 1:length(list)) {
  test <- ts(data_clean[,list[i]], frequency = 12, start = c(2018,1))
  hw1 <- hw(test,seasonal="additive",h=hor,level=95)
  hw1_df <-as.data.frame(hw1)
  fc_hw1_point <- as.data.frame(hw1_df$"Point Forecast")
  colnames(fc_hw1_point)[1] <- paste(list[i])
  fc_hw1 <- cbind(fc_hw1_point, fc_hw1)
}
#--------------------- HOLT WINTER DAMPED ---------------------#
fc_hw3 <- data.frame(matrix(nrow = hor, ncol = 0))
for (i in 1:length(list)) {
  test <- ts(data_clean[,list[i]], frequency = 12, start = c(2018,1))
  hw3 <- hw(test,seasonal="additive", damped = T,h=hor,level=95)
  hw3_df <-as.data.frame(hw3)
  fc_hw3_point <- as.data.frame(hw3_df$"Point Forecast")
  colnames(fc_hw3_point)[1] <- paste(list[i])
  fc_hw3 <- cbind(fc_hw3_point, fc_hw3)
}

#--------------------- AUTO ARIMA ---------------------#
fc_arima <- data.frame(matrix(nrow = hor, ncol = 0))
for (i in 1:length(list)) {
  test <- ts(data_clean[,list[i]], frequency = 12, start = c(2018,1))
  pf_auto.arima <- auto.arima(test)
  pf_auto.arima_forecast <- forecast(pf_auto.arima,h=hor)
  pf_auto.arima_df <-as.data.frame(pf_auto.arima_forecast)
  fc_auto.arima_point <- as.data.frame(pf_auto.arima_df$"Point Forecast")
  colnames(fc_auto.arima_point)[1] <- paste(list[i])
  fc_arima <- cbind(fc_auto.arima_point, fc_arima)
}

#--------------------- ETS ---------------------#
fc_ets <- data.frame(matrix(nrow = hor, ncol = 0))
for (i in 1:length(list)) {
  test <- ts(data_clean[,list[i]], frequency = 12, start = c(2018,1))
  test_ets <- ets(test,model="ZZZ",
                  damped=NULL,
                  alpha=NULL,
                  beta=NULL,
                  gamma=NULL,
                  phi=NULL,
                  additive.only=FALSE,
                  lambda=NULL,
                  lower=c(rep(0.0001,3), 0.8),
                  upper=c(rep(0.9999,3),0.98),
                  opt.crit=c("lik","amse","mse","sigma","mae"),
                  nmse=3, bounds=c("both","usual","admissible"),
                  ic=c("aicc","aic","bic"), restrict=TRUE)
  test_ets_forecast <- forecast(test_ets,h=hor)
  test_ets_df <- as.data.frame(test_ets_forecast) 
  test_ets_Point <- as.data.frame(test_ets_df$"Point Forecast")
  colnames(test_ets_Point)[1] <- paste(list[i])
  fc_ets <- cbind(test_ets_Point, fc_ets)
}

#--------------------- TONG HOP ---------------------#
last_date <- max(ymd(paste0(data1$YEAR_MONTH, "-01"))) %m+% months(1) 
fc_date <- format(as.Date(paste0(last_date, "-01")), "%Y-%m-01")

fc_ets$MODEL <- 'ETS'
fc_arima$MODEL <- 'SARIMA'
fc_hw1$MODEL <- 'HW1'
fc_hw3$MODEL <- 'HW3'


fc_ets$MONTH <- seq(as.Date(fc_date), by = '1 month', length.out = nrow(fc_ets))
fc_arima$MONTH <- seq(as.Date(fc_date), by = '1 month', length.out = nrow(fc_arima))
fc_hw1$MONTH <- seq(as.Date(fc_date), by = '1 month', length.out = nrow(fc_hw1))
fc_hw3$MONTH <- seq(as.Date(fc_date), by = '1 month', length.out = nrow(fc_hw3))

forecast <- rbind(fc_hw1, fc_hw3,fc_arima,fc_ets)


forecast <- melt(forecast, id = c("MONTH","MODEL"))
forecast$REGION <- substr(forecast$variable,1,3)
forecast$PFSAP <- substr(forecast$variable,5,length(forecast$variable))
forecast$MKDATE <- last_date    
forecast <- rename(forecast, QNT = value)
forecast <- subset(forecast, select = c(MONTH,MODEL,QNT,REGION,PFSAP,MKDATE))
forecast$QNT <- as.integer(forecast$QNT)
forecast <- subset(forecast, QNT >0   )

#--------------------- INSERT ROW TO BIGQUERY ---------------------#
bq_table_upload(x="pnj-sc-aa-enhance.OUTPUT_FORECAST.FCT_MODEL", 
                values= forecast, 
                create_disposition='CREATE_NEVER', 
                write_disposition='WRITE_APPEND')
