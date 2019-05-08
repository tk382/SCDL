tae_denoise = function(text.file.name, 
                       is.large.data = F,
                       ncores = 1, 
                       verbose = F, 
                       batch_size = NULL, 
                       clearup.python.session = T, 
                       seed = 1, 
                       fold = 6, 
                       samp = 3, 
                       epsilon = 1e-10, 
                       nonmissing_indicator = 1,
                       ...){

  
  write_output_to_tsv = is.large.data
  sctransfer_tae <- reticulate::import("sctransfer_tae")
  main <- reticulate::import_main(convert = F)
  print("Python module sctransfer_tae imported ...")
  
  set.seed(seed)
  
  # format <- strsplit(text.file.name, '[.]')[[1]]
  # format <- paste(".", format[length(format)], sep = "")
  out_dir <- strsplit(text.file.name, split = "/")[[1]]
  out_dir <- paste(out_dir[-length(out_dir)], collapse = "/")
  data = readRDS(text.file.name)
  # data <- readRDS(gsub(format, "_temp.rds", text.file.name))
  if (is.null(batch_size)){
    batch_size <- as.integer(max(ncol(data) / 50, 32))
  } else{
    batch_size <- as.integer(batch_size)
  }
  
  x = data
  
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
    
    test.x = x[, test.idx]
    
    #train using train.idx, but measure MSE with the test set
    api <- sctransfer_tae$api
    
    gnames <- rownames(x[, train.idx])
    cnames <- colnames(x[, train.idx])
    x[, train.idx] <- Matrix::Matrix(x[, train.idx], sparse = T)
    mtx_file <- paste0(out_dir, "/SAVERX_temp.mtx")
    Matrix::writeMM(x[, train.idx], file = mtx_file)
    
    nonmissing_indicator <- api$np$asarray(nonmissing_indicator) 
    gc()
    
    gnames <- rownames(test.x)
    cnames <- colnames(test.x)
    test.x <- Matrix::Matrix(test.x, sparse = T)
    test_mtx_file <- paste0(out_dir, "/SAVERX_temp_test.mtx")
    Matrix::writeMM(test.x, file = test_mtx_file)
    rm(test.x)
    gc()
    main$result <- api$autoencode(mtx_file = mtx_file,
                                  pred_mtx_file = test_mtx_file,
                                  nonmissing_indicator = nonmissing_indicator,                      
                                  out_dir = out_dir,
                                  batch_size = batch_size, 
                                  write_output_to_tsv = write_output_to_tsv,
                                  ...)
    if (!write_output_to_tsv) {
      x.autoencoder <- t(reticulate::py_to_r(main$result$obsm[['X_dca']]))
      colnames(x.autoencoder) <- cnames
      rownames(x.autoencoder) <- gnames
    } else{
      x.autoencoder <- NULL
    }
    reticulate::py_run_string("
del result
import gc
gc.collect()")
    
    if (write_output_to_tsv) {
      x.autoencoder <- t(as.matrix(data.table::fread(paste0(out_dir, 
                                                            "/SAVERX_temp_pred_mean_norm.tsv"), header = F)))
    }
    
    #estimate mu: gene mean after normalizing library size for this training index
    est.mu <- Matrix::rowMeans(Matrix::t(Matrix::t(x[, train.idx]) / Matrix::colSums(x[, train.idx])) * 10000)
    #normalize the library size of the denoised matrix for this training index
    est.autoencoder <- Matrix::t(Matrix::t(x.autoencoder) * Matrix::colSums(test.x)) / 10000 
    rm(x.autoencoder)
    gc()
    est.const <- est.mu %*% Matrix::t(Matrix::colSums(test.x)) / 10000  #intercept part..?
    
    err1 <- -Matrix::rowMeans(test.x * log(est.autoencoder + epsilon) - est.autoencoder)  # poisson loss
    err2 <- -Matrix::rowMeans(test.x * log(est.const + epsilon) - est.const)  # what is this
    
    #combine across the 3-way split
    err.autoencoder <- err.autoencoder + err1  
    err.const <- err.const + err2
    rm(test.x, est.mu, est.autoencoder, est.const, err1, err2)
    gc()
  }
  
  est.mu <- Matrix::rowMeans(Matrix::t(Matrix::t(x) / Matrix::colSums(x)) * 10000)
  #  est.const <- est.mu %*% t(rep(1, n.cell))
  gnames <- rownames(x)
  cnames <- colnames(x)
  
  print("Final prediction round using all cells. See below the summary of the autoencoder model:")
  test.x = NULL
  test_mtx_file <- NULL
  pred_mtx_file = NULL
  main$result <- api$autoencode(mtx_file = x,
                                pred_mtx_file = NULL,
                                nonmissing_indicator = nonmissing_indicator,                      
                                out_dir = out_dir,
                                batch_size = batch_size, 
                                write_output_to_tsv = write_output_to_tsv,
                                ...)
  if (!write_output_to_tsv) {
    x.autoencoder <- t(reticulate::py_to_r(main$result$obsm[['X_dca']]))
    colnames(x.autoencoder) <- cnames
    rownames(x.autoencoder) <- gnames
  } else{
    x.autoencoder <- NULL
  }
  reticulate::py_run_string("
del result
import gc
gc.collect()")
  
  if (write_output_to_tsv) {
    x.autoencoder <- t(as.matrix(data.table::fread(paste0(out_dir, 
                                                          "/SAVERX_temp_pred_mean_norm.tsv"), header = F)))
  }
  
  if (write_output_to_tsv) {
    x.autoencoder = t(as.matrix(data.table::fread(paste0(out_dir, "/SAVERX_temp_mean_norm.tsv"), header = F)))
    rownames(x.autoencoder) = gnames
    colnames(x.autoencoder) = cnames
  }
  
  x.autoencoder[err.autoencoder - err.const > 0, ] = est.mu[err.autoencoder - err.const > 0]
  
  return(list(x.autoencoder = x.autoencoder, 
              err.autoencoder = err.autoencoder, 
              err.const = err.const))
  
}