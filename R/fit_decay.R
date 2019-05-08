get_positive_mean = function(x){
  ind = which( x > 0)
  if(length(ind) > 0){
    mean(x[ind])
  }else{
    return(0)
  }
}

fit_decay = function(orig, cutoff = 99){
  #orig must be gene by cell
  library(Matrix)
  library(reshape2)
  library(dplyr)
  library(broom)
  dim(orig)
  dropout_rate = rowSums(orig==0) / ncol(orig)
  gene_mean = apply(orig, 1, get_positive_mean)
  par(mfrow = c(1,2))
  plot(dropout_rate ~ gene_mean, cex = 0.3, main = 'drop out ~ mean(exp)')
  
  ind = which(dropout_rate < 0.95 & dropout_rate > 0.05)
  gene_mean2 = gene_mean[ind]
  dropout_rate2 = dropout_rate[ind]
  
  distdf = data.frame(gene_mean = gene_mean2, dropout_rate = dropout_rate2)
  distdf$range = cut(distdf$dropout_rate, seq(0.01,0.99,length=cutoff), include.lowest=TRUE)
  distdf = distdf %>% group_by(range) %>% summarize(var = var(gene_mean), mean = mean(gene_mean))
  distdf$range = as.character(distdf$range)
  distdf$minrange = substring(sapply(strsplit(distdf$range, ','), function(x) x[1]), 2)
  distdf$maxrange = gsub('.{1}$', '', sapply(strsplit(distdf$range, ','), function(x) x[2]) )
  distdf$minrange = as.numeric(distdf$minrange)
  distdf$maxrange = as.numeric(distdf$maxrange)
  distdf$mean_dropout_rate = (distdf$minrange + distdf$maxrange) / 2
  
  mod = nls(mean_dropout_rate ~ SSasymp(mean, yf, y0, log_alpha), data = distdf)
  
  plot(distdf$mean_dropout_rate ~ distdf$mean, cex = 0.3, ylim = c(-0.1,1), main = "E(gene mean | dropout rate)"); 
  points(fitted(mod) ~ distdf$mean, col = 'red', cex = 0.3); 
  legend('topright', col = c('black', 'red'), pch = 1, legend = c('observed', 'fitted'))
  
  
  return(broom::tidy(mod))
}
