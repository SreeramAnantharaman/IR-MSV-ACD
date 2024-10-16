set.seed(as.integer(1234))

library(cmdstanr)
check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)
library(posterior)
library(Metrics)
library(loo)

file <- "irbmsv_acd_lasso.stan"

# Compile the model with the specified Stan version
model <- cmdstan_model(stan_file = file,cpp_options = list(stan_threads = TRUE))

##################################
# Simulation:
##################################

# Simulation parameters:
m=1
T=5020 # Number of time points
p=4 # dimension of stock
q=2 # Lag order for g
k=q+1

# Parameter values for simulations
delta <- 2.2
omega <- 0.1
alpha <- c(0.7,0.2)
psi_sim <- rep(NA,T)
g_sim <- rep(NA,T)
y=array(0,c(m,T,p))
Sigma <- matrix(1,nrow = p,ncol=p)

g <- rep(NA,T)
y=array(0,c(m,T,p))
Sigma <- matrix(1,nrow = p,ncol=p)

require(MASS)

for (j in 1:m){
  mu=c(-9,-9.5,-8.5,-8)
  itau2=c(0.9,0.7,0.5,0.3) #sigmasq
  phi=c(0.7,0.5,0.3,0.9)
  h=matrix(0,T,p)
  sigma_y=array(0,c(T,p,p))
  h[1,]=rnorm(p,mean=mu,sd=sqrt(itau2/(1-phi^2)))
  rho <- c(0.3,0.25,0.35,0.35,0.40,0.40)
  cor_matrix <- diag(p)
  cor_matrix[lower.tri(cor_matrix)] <- rho
  cor_matrix[upper.tri(cor_matrix)] <- cor_matrix[lower.tri(cor_matrix)]
  Sigma[1:p,1:p] <- cor_matrix
  
  for (t in 1:q){
    psi_sim[t] <- omega
    g_sim[t] <- rgamma(n=1, shape=delta, scale=psi_sim[t]/delta)
  }
  for (t in (q+1):(T)){
    psi_sim[t] <- omega+alpha[1:q]%*%g_sim[(t-1):(t-q)]
    g_sim[t] <- rgamma(n=1, shape=delta, scale=psi_sim[t]/delta)
  }
  
  sigma_y[1,,]=diag(exp(h[1,]/2))%*%Sigma%*%diag(exp(h[1,]/2))
  y[j,1,]=mvrnorm(n = 1, rep(0,p),sigma_y[1,,])
  for (i in 2:T){
    h[i,]=rnorm(p,mean=mu+phi^(g_sim[i])*(h[i-1,]-mu),sd=sqrt(itau2*(1-phi^(2*g_sim[i]))/(1-phi^2)))
    sigma_y[i,,]=diag(exp(h[i,]/2))%*%Sigma%*%diag(exp(h[i,]/2))
    y[j,i,]=mvrnorm(n = 1, rep(0,p),sigma_y[i,,] )
  }
}

#######################
# Stan estimation:
#######################

y <- data.frame(y1=y[,,1],y2=y[,,2],y3=y[,,3],y4=y[,,4])
y=as.matrix(y)
g <- data.frame(g_sim)
g=as.numeric(g[,1])

data_list <- list(
  T = T-20,
  p = p,
  q = 10,
  k = 11,
  g = g[1:5000],
  y = y[1:5000,],
  #alpha_dir=0.5,
  sds=rep(1,p),
  mean_y=matrix(0,nrow=1,ncol=p))

## HMC
options(mc.cores=4) # choose the number of cores you want to run the model on
fit=model$sample(data=data_list,seed=1,chains = 4,iter_sampling = 1000,parallel_chains = 4,threads_per_chain = 12,iter_warmup=1000,thin=4)
par_summary=fit$summary(variables = c("Lcorr", "tau[1]", "tau[2]"
                                      ,"tau[3]", "tau[4]","phi[1]", "phi[2]","phi[3]","phi[4]","delta",
                                      "omega","durpar[1]", "durpar[2]","durpar[3]", "durpar[4]"
                                  ,"durpar[5]", "durpar[6]","durpar[7]", "durpar[8]","durpar[9]", "durpar[10]"
                                      ,"mu[1]", "mu[2]", "mu[3]","mu[4]","alpha_dir"))

summary=cbind(par_summary,fit$summary(c("Lcorr", "tau[1]", "tau[2]"
                                        ,"tau[3]", "tau[4]","phi[1]", "phi[2]","phi[3]","phi[4]","delta",
                                        "omega","durpar[1]", "durpar[2]","durpar[3]", "durpar[4]"
                                  ,"durpar[5]", "durpar[6]","durpar[7]", "durpar[8]","durpar[9]", "durpar[10]"
                                        ,"mu[1]", "mu[2]", "mu[3]","mu[4]","alpha_dir"), quantile, .args = list(probs = c(0.025, .975)))[,c(2,3)])
summary

samples <- fit$draws(variables = c("Lcorr", "tau[1]", "tau[2]"
                                   ,"tau[3]", "tau[4]","phi[1]", "phi[2]","phi[3]","phi[4]","delta",
                                   "omega","durpar[1]", "durpar[2]","durpar[3]", "durpar[4]"
                                  ,"durpar[5]", "durpar[6]","durpar[7]", "durpar[8]","durpar[9]", "durpar[10]"
                                   ,"mu[1]", "mu[2]", "mu[3]","mu[4]","alpha_dir"),format="df")
fit$diagnostic_summary()
nSamp <- nrow(samples)
lpd=as.matrix(fit$lp(),nrow=nSamp,ncol=1)
waic_est=waic(lpd)

filenamet<-sprintf("bmsv_g2lasso_time.rds")
filename0 <-sprintf("bmsv_g2lasso_lpd.rds")
filename <-sprintf("bmsv_g2lasso_summ.rds")
filename1 <-sprintf("bmsv_g2lasso_full.rds")
filename2 <- sprintf("bmsv_g2lasso_waic.rds")

saveRDS(object = summary, file = filename)
saveRDS(object = samples, file = filename1)
saveRDS(object = waic_est, file = filename2)
saveRDS(object = lpd, file = filename0)
saveRDS(fit$time()$total, file = filenamet)

###########################################################################
## Fitting
###########################################################################

nSamp <- nrow(samples)
n <- 5000
p=4
q=10

eta_fit <- fit$draws("delta")
omega_fit <- fit$draws("omega")
alpha_fit <- fit$draws(c("durpar[1]", "durpar[2]","durpar[3]", "durpar[4]"
                         ,"durpar[5]", "durpar[6]","durpar[7]", "durpar[8]","durpar[9]", "durpar[10]"),format = "df")
phi_fit <- fit$draws(c("phi[1]","phi[2]","phi[3]","phi[4]"))
itau_fit <- fit$draws(c("tau[1]","tau[2]","tau[3]","tau[4]"))
mu_fit <- fit$draws(c("mu[1]","mu[2]","mu[3]","mu[4]"))
rho_fit <- fit$draws(c("Lcorr[1,2]","Lcorr[1,3]","Lcorr[1,4]","Lcorr[2,3]","Lcorr[2,4]","Lcorr[3,4]"))

psi_fit <- rep(NA,n)
g_fit <- rep(NA,n)
Sigma_fit <- matrix(1,nrow = p,ncol=p)
y_fit=array(0,c(m,n,p))

gppSamples <- matrix(0, nSamp, n)
ppSamples1 <- matrix(0, nSamp, n)
ppSamples2 <- matrix(0, nSamp, n)
ppSamples3 <- matrix(0, nSamp, n)
ppSamples4 <- matrix(0, nSamp, n)
psifits <- matrix(0, nSamp, n)
lt_st1 <- matrix(0, nSamp, n)
lt_st2 <- matrix(0, nSamp, n)
lt_st3 <- matrix(0, nSamp, n)
lt_st4 <- matrix(0, nSamp, n)

require(MASS)

for (id in 1:nSamp){
  eta <- eta_fit[id]
  omega <- omega_fit[id]
  alpha <- as.numeric(alpha_fit[id,1:q])
  mu <- mu_fit[id]
  itau2 <- itau_fit[id]
  phi <- phi_fit[id]
  rho <- rho_fit[id]
  T=5000
  
  h_fit=matrix(0,T,p)
  sigma_y=array(0,c(T,p,p))
  h_fit[1,]=rnorm(p,mean=mu,sd=sqrt(itau2/(1-phi^2)))
  cor_matrix <- diag(p)
  cor_matrix[lower.tri(cor_matrix)] <- rho
  cor_matrix[upper.tri(cor_matrix)] <- cor_matrix[lower.tri(cor_matrix)]
  Sigma_fit[1:p,1:p] <- cor_matrix
  
  for (t in 1:q){
    psi_fit[t] <- omega
    g_fit[t] <- rgamma(n=1, shape=eta, scale=psi_fit[t]/eta)
  }
  for (t in (q+1):(T)){
    psi_fit[t] <- omega+alpha[1:q]%*%g[(t-1):(t-q)]
    g_fit[t] <- rgamma(n=1, shape=eta, scale=psi_fit[t]/eta)
  }
  
  sigma_y[1,,]=diag(exp(h_fit[1,]/2))%*%Sigma_fit%*%diag(exp(h_fit[1,]/2))
  y_fit[j,1,]=mvrnorm(n = 1, rep(0,p),sigma_y[1,,])
  for (i in 2:T){
    h_fit[i,]=rnorm(p,mean=mu+phi^(g_fit[i])*(h[i-1,]-mu),sd=sqrt(itau2*(1-phi^(2*g_fit[i]))/(1-phi^2)))
    sigma_y[i,,]=diag(exp(h_fit[i,]/2))%*%Sigma_fit%*%diag(exp(h_fit[i,]/2))
    y_fit[j,i,]=mvrnorm(n = 1, rep(0,p),sigma_y[i,,])
  }
  gppSamples[id,] <- g_fit
  psifits[id,] <- psi_fit
  lt_st1[id,] <- h_fit[,1]
  lt_st2[id,] <- h_fit[,2]
  lt_st3[id,] <- h_fit[,3]
  lt_st4[id,] <- h_fit[,4]
  ppSamples1[id,] <- y_fit[,,1]
  ppSamples2[id,] <- y_fit[,,2]
  ppSamples3[id,] <- y_fit[,,3]
  ppSamples4[id,] <- y_fit[,,4]
}

############################################################################
## psi fits

psi_mae_train11=rep(NA,nSamp)
for (i in 1: nSamp){
  psi_mae_train11[i]=mae(psi_sim[1:5000], as.numeric(psifits[i,]))
}
psi_mae1_train11=mean(psi_mae_train11)

###############################################################################
## gaps fits

gaps_mae_train11=rep(NA,nSamp)
for (i in 1: nSamp){
  gaps_mae_train11[i]=mae(g[1:5000], as.numeric(gppSamples[i,]))
}
gaps_mae1_train11=mean(gaps_mae_train11)

############################################################################
## latent state fits

lt_mae_train11=rep(NA,nSamp)
lt_mae_train22=rep(NA,nSamp)
lt_mae_train33=rep(NA,nSamp)
lt_mae_train44=rep(NA,nSamp)
for (i in 1: nSamp){
  lt_mae_train11[i]=mae(h[1:5000,1], as.numeric(lt_st1[i,]))
  lt_mae_train22[i]=mae(h[1:5000,2], as.numeric(lt_st2[i,]))
  lt_mae_train33[i]=mae(h[1:5000,3], as.numeric(lt_st3[i,]))
  lt_mae_train44[i]=mae(h[1:5000,4], as.numeric(lt_st4[i,]))
}
lt_mae1_train11=mean(lt_mae_train11)
lt_mae1_train22=mean(lt_mae_train22)
lt_mae1_train33=mean(lt_mae_train33)
lt_mae1_train44=mean(lt_mae_train44)

############################################################################
## log-returns fits

rt_mae_train11=rep(NA,nSamp)
rt_mae_train22=rep(NA,nSamp)
rt_mae_train33=rep(NA,nSamp)
rt_mae_train44=rep(NA,nSamp)
for (i in 1: nSamp){
  rt_mae_train11[i]=mae(y[1:5000,1], as.numeric(ppSamples1[i,]))
  rt_mae_train22[i]=mae(y[1:5000,2], as.numeric(ppSamples2[i,]))
  rt_mae_train33[i]=mae(y[1:5000,3], as.numeric(ppSamples3[i,]))
  rt_mae_train44[i]=mae(y[1:5000,4], as.numeric(ppSamples4[i,]))
}
rt_mae1_train11=mean(rt_mae_train11)
rt_mae1_train22=mean(rt_mae_train22)
rt_mae1_train33=mean(rt_mae_train33)
rt_mae1_train44=mean(rt_mae_train44)
rt_mae1_train44

############################################################################
## Combine MAE

mae1_train=cbind(psi_mae1_train11,gaps_mae1_train11, 
                 lt_mae1_train11, lt_mae1_train22, lt_mae1_train33,lt_mae1_train44,
                 rt_mae1_train11,rt_mae1_train22,rt_mae1_train33,rt_mae1_train44)

filename3_1 <- sprintf("bmsv_g2lasso_maetrain1.rds")
saveRDS(object = mae1_train, file = filename3_1)

###########################################################################
## Forecasting
###########################################################################

m=1
n=20
p=4
q=10

psi_fore <- rep(NA,n)
gppSamples_fore <- rep(NA,n)
Sigma_fore <- matrix(1,nrow = p,ncol=p)
y_fore=array(0,c(m,n,p))

gaps_fore <- matrix(0, nSamp, n)
ppSamples1_fore <- matrix(0, nSamp, n)
ppSamples2_fore <- matrix(0, nSamp, n)
ppSamples3_fore <- matrix(0, nSamp, n)
ppSamples4_fore <- matrix(0, nSamp, n)
psij_fore <- matrix(0, nSamp, n)
lt_st1_fore <- matrix(0, nSamp, n)
lt_st2_fore <- matrix(0, nSamp, n)
lt_st3_fore <- matrix(0, nSamp, n)
lt_st4_fore <- matrix(0, nSamp, n)

for (id in 1:nSamp){
  eta <- eta_fit[id]
  omega <- omega_fit[id]
  alpha <- as.numeric(alpha_fit[id,1:q])
  mu <- mu_fit[id]
  itau2 <- itau_fit[id]
  phi <- phi_fit[id]
  rho <- rho_fit[id]
  T=20
  g_val=g[1:5000]
  
  h_fore=matrix(0,T,p)
  sigma_y=array(0,c(T,p,p))
  cor_matrix <- diag(p)
  cor_matrix[lower.tri(cor_matrix)] <- rho
  cor_matrix[upper.tri(cor_matrix)] <- cor_matrix[lower.tri(cor_matrix)]
  Sigma_fore[1:p,1:p] <- cor_matrix
  
  for (t in 1:q){
    psi_fore[t] <- omega+alpha[1:q]%*%g_val[(5000+t-1):(5000+t-q)]
    gppSamples_fore[t] <- rgamma(n=1, shape=eta, scale=psi_fore[t]/eta)
    g_val[5000+t]=gppSamples_fore[t]
  }
  for (t in (q+1):(T)){
    psi_fore[t] <- omega+alpha[1:q]%*%gppSamples_fore[(t-1):(t-q)]
    gppSamples_fore[t] <- rgamma(n=1, shape=eta, scale=psi_fore[t]/eta)
  }
  h_fore[1,]=rnorm(p,mean=mu+phi^(gppSamples_fore[1])*(h[5000,]-mu),sd=sqrt(itau2*(1-phi^(2*gppSamples_fore[1]))/(1-phi^2)))
  sigma_y[1,,]=diag(exp(h_fore[1,]/2))%*%Sigma_fore%*%diag(exp(h_fore[1,]/2))
  y_fore[j,1,]=mvrnorm(n = 1, rep(0,p),sigma_y[1,,])
  for (i in 2:T){
    h_fore[i,]=rnorm(p,mean=mu+phi^(gppSamples_fore[i])*(h_fore[i-1,]-mu),sd=sqrt(itau2*(1-phi^(2*gppSamples_fore[i]))/(1-phi^2)))
    sigma_y[i,,]=diag(exp(h_fore[i,]/2))%*%Sigma_fore%*%diag(exp(h_fore[i,]/2))
    y_fore[j,i,]=mvrnorm(n = 1, rep(0,p),sigma_y[i,,])
  }
  gaps_fore[id,] <- gppSamples_fore
  psij_fore[id,] <- psi_fore
  lt_st1_fore[id,] <- h_fore[,1]
  lt_st2_fore[id,] <- h_fore[,2]
  lt_st3_fore[id,] <- h_fore[,3]
  lt_st4_fore[id,] <- h_fore[,4]
  ppSamples1_fore[id,] <- y_fore[,,1]
  ppSamples2_fore[id,] <- y_fore[,,2]
  ppSamples3_fore[id,] <- y_fore[,,3]
  ppSamples4_fore[id,] <- y_fore[,,4]
}

#############################################################################
## psi forecasts

psi_mae_test11=rep(NA,nSamp)
for (i in 1: nSamp){
  psi_mae_test11[i]=mae(psi_sim[5001:5020], as.numeric(psij_fore[i,]))
}
psi_mae1_test11=mean(psi_mae_test11)

#############################################################################
## gaps forecasts

gaps_mae_test11=rep(NA,nSamp)
for (i in 1: nSamp){
  gaps_mae_test11[i]=mae(g[5001:5020], as.numeric(gaps_fore[i,]))
}
gaps_mae1_test11=mean(gaps_mae_test11)

################################################################
## ht forecasts

lt_mae_test11=rep(NA,nSamp)
lt_mae_test22=rep(NA,nSamp)
lt_mae_test33=rep(NA,nSamp)
lt_mae_test44=rep(NA,nSamp)
for (i in 1: nSamp){
  lt_mae_test11[i]=mae(h[5001:5020,1], as.numeric(lt_st1[i,]))
  lt_mae_test22[i]=mae(h[5001:5020,2], as.numeric(lt_st2[i,]))
  lt_mae_test33[i]=mae(h[5001:5020,3], as.numeric(lt_st3[i,]))
  lt_mae_test44[i]=mae(h[5001:5020,4], as.numeric(lt_st4[i,]))
}
lt_mae1_test11=mean(lt_mae_test11)
lt_mae1_test22=mean(lt_mae_test22)
lt_mae1_test33=mean(lt_mae_test33)
lt_mae1_test44=mean(lt_mae_test44)

############################################################################
## log-returns forecasts

rt_mae_test11=rep(NA,nSamp)
rt_mae_test22=rep(NA,nSamp)
rt_mae_test33=rep(NA,nSamp)
rt_mae_test44=rep(NA,nSamp)
for (i in 1: nSamp){
  rt_mae_test11[i]=mae(y[5001:5020,1], as.numeric(ppSamples1_fore[i,]))
  rt_mae_test22[i]=mae(y[5001:5020,2], as.numeric(ppSamples2_fore[i,]))
  rt_mae_test33[i]=mae(y[5001:5020,3], as.numeric(ppSamples3_fore[i,]))
  rt_mae_test44[i]=mae(y[5001:5020,4], as.numeric(ppSamples4_fore[i,]))
}
rt_mae1_test11=mean(rt_mae_test11)
rt_mae1_test22=mean(rt_mae_test22)
rt_mae1_test33=mean(rt_mae_test33)
rt_mae1_test44=mean(rt_mae_test44)

############################################################################
## Combine MAE
mae1_test=cbind(psi_mae1_test11, gaps_mae1_test11, lt_mae1_test11, 
                lt_mae1_test22,lt_mae1_test33,lt_mae1_test44,rt_mae1_test11,rt_mae1_test22,rt_mae1_test33,
                rt_mae1_test44)

filename5_1 <- sprintf("bmsv_g2lasso_maetest1.rds")
saveRDS(object = mae1_test, file = filename5_1)
