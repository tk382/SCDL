#' Cross-validation filtering of autoencoder
#'
#' Cross-validation is done to determine which genes can not be predicted well, by comparing the autoencoder predicted loss with the loss estimating the gene expression as a constant across cells
#'
#' @inheritParams autoencode
#' @param fold Number of total CV folds
#' @param samp Number of sampled folds taken to reduce CV time cost
#' @return a list of the filtered predicted data matrix and the CV error
#'
#' @export

autoFilterCV <- function(x,
                         curve_file_name,
                         python.module,
                         main,
                         nonmissing_indicator = 1,
                         out_dir = ".",
                         batch_size = 32L,
                         write_output_to_tsv = F,
                         fold = 6, samp = 3, epsilon = 1e-10, seed = 1, ...) {
  
  print(out_dir)

  set.seed(seed)

  n.cell <- ncol(x)
  n.gene <- nrow(x)

  idx.perm <- sample(1:n.cell, n.cell)

  n.test <- floor(n.cell/fold)
  
  err.autoencoder <- err.const <- rep(0, n.gene)
  ss <- sample(1:fold, samp)
  for (k in 1:samp) {
    print(paste("Cross-validation round:", k))
    i = ss[k]
    train.idx = idx.perm[-(((i-1)*n.test + 1):(i * n.test))]
    test.idx = idx.perm[((i-1)*n.test + 1):(i * n.test)]

    x.test = x[, test.idx]
    x.train = x[, train.idx]
    x.autoencoder = autoencode(x.train,  
                                curve_file_name,
                                python.module,
                                main,
                                x.test,
                                nonmissing_indicator,
                                out_dir,
                                batch_size, 
                                write_output_to_tsv,
                                verbose_sum = F, verbose_fit = 0L)

    if (write_output_to_tsv) {
      x.autoencoder <- t(as.matrix(data.table::fread(paste0(out_dir, 
                                                          "/SAVERX_temp_pred_mean_norm.tsv"), header = F)))
    }

    est.autoencoder = x.autoencoder$result #32738 x 2179
    rm(x.autoencoder)
    gc()
    
    est.mu = rowMeans(x[,train.idx]) #32738
    est.const = est.mu %*% t(rep(1, length(test.idx)))
    
    #likelihood, not error
    err1 = rowMeans(dpois(as.matrix(x.test), as.matrix(est.autoencoder), log=TRUE)) 
    err2 = rowMeans(dpois(as.matrix(x.test), as.matrix(est.const), log=TRUE))

    err.autoencoder <- err.autoencoder + err1  
    err.const <- err.const + err2
    rm(x.test, est.mu, est.autoencoder, est.const, err1, err2)
    gc()
  }

  est.mu = rowMeans(x)
  est.const = est.mu %*% t(rep(1, ncol(x)))
  gnames <- rownames(x)
  cnames <- colnames(x)

  print("Final prediction round using all cells. See below the summary of the autoencoder model:")
  x.autoencoder <- autoencode(x,  
                              curve_file_name,
                              python.module,
                              main,
                              NULL,
                              nonmissing_indicator,
                              out_dir,
                              batch_size, 
                              write_output_to_tsv, 
                              verbose_fit = 0L, ...)
  rm(x)
  gc()

  if (write_output_to_tsv) {
    x.autoencoder$result <- t(as.matrix(data.table::fread(paste0(out_dir, 
                                                        "/SAVERX_temp_mean_norm.tsv"), header = F)))
    rownames(x.autoencoder$result) <- gnames
    colnames(x.autoencoder$result) <- cnames
  }
  
  #if autoencder prediction is worse than the mean prediction, replace it mean
  x.autoencoder$result[err.autoencoder < err.const, ] <- est.mu[err.autoencoder < err.const]


  return(list(x.autoencoder = x.autoencoder, 
              err.autoencoder = err.autoencoder, 
              err.const = err.const))
}
