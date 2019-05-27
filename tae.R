setwd("~/Work/SC/SCDL_ver4")
R.utils::sourceDirectory("R/", modifiedOnly = FALSE)
label = read.table("../data/10X_pbmc_filtered/truelabel.tsv")$V1
text.file.name = "../data/10X_pbmc_filtered/matrix.rds"
sctransfer_tae <- reticulate::import("pycodes")
format <- strsplit(text.file.name, '[.]')[[1]]
format <- paste(".", format[length(format)], sep = "")
main <- reticulate::import_main(convert = F)
out_dir <- strsplit(text.file.name, split = "/")[[1]]
out_dir <- paste(out_dir[-length(out_dir)], collapse = "/")
write_output_to_tsv <- F
data = readRDS(text.file.name)
batch_size <- as.integer(max(ncol(data) / 50, 32))
# curve = fit_decay(data)
curve_file_name = gsub(format, "_curve.txt", text.file.name)
# write.table(curve$estimate, curve_file_name, col.names=FALSE, row.names=FALSE)
seed=1
set.seed(seed)
x = data; rm(data); gc()
n.cell <- ncol(x)
n.gene <- nrow(x)
fold=6; samp=3; epsilon = 1e-10
idx.perm <- sample(1:n.cell, n.cell)
n.test <- floor(n.cell/fold)
err.autoencoder <- err.const <- rep(0, n.gene)
ss <- sample(1:fold, samp)
k=1
i = ss[k]
train.idx = idx.perm[-(((i-1)*n.test + 1):(i * n.test))]
test.idx = idx.perm[((i-1)*n.test + 1):(i * n.test)]
test.x = x[, test.idx]
train.x = x[, train.idx]
api <- sctransfer_tae$api
gnames <- rownames(train.x)
cnames <- colnames(train.x)
train.x <- Matrix::Matrix(train.x, sparse = T)
mtx_file <- paste0(out_dir, "/SAVERX_temp.mtx")
Matrix::writeMM(train.x, file = mtx_file)
gnames = rownames(test.x)
cnames = colnames(test.x)
test.x = Matrix::Matrix(test.x, sparse = T)
test_mtx_file = paste0(out_dir, "/SAVERX_temp_test.mtx")
Matrix::writeMM(test.x, file = test_mtx_file)
pred_mtx_file = test_mtx_file
nonmissing_indicator = 1
tmp = api$autoencode(mtx_file = mtx_file,
                     curve_file_name = curve_file_name,
                     pred_mtx_file = test_mtx_file,
                     nonmissing_indicator = 1,                      
                     out_dir = out_dir,
                     batch_size = batch_size, 
                     write_output_to_tsv = write_output_to_tsv)
est.autoencoder = t(tmp[[1]]) #32738 x 2179
est.mu = rowMeans(train.x) #32738
est.const = est.mu %*% t(rep(1, length(test.idx)))
err1 = rowMeans(dpois(as.matrix(test.x), as.matrix(est.autoencoder), log=TRUE))
err2 = rowMeans(dpois(as.matrix(test.x), as.matrix(est.const), log=TRUE))
err.autoencoder <- err.autoencoder + err1  
err.const <- err.const + err2


k=2
i = ss[k]
train.idx = idx.perm[-(((i-1)*n.test + 1):(i * n.test))]
test.idx = idx.perm[((i-1)*n.test + 1):(i * n.test)]
test.x = x[, test.idx]
train.x = x[, train.idx]
gnames <- rownames(train.x)
cnames <- colnames(train.x)
train.x <- Matrix::Matrix(train.x, sparse = T)
mtx_file <- paste0(out_dir, "/SAVERX_temp.mtx")
Matrix::writeMM(train.x, file = mtx_file)
gnames = rownames(test.x)
cnames = colnames(test.x)
test.x = Matrix::Matrix(test.x, sparse = T)
test_mtx_file = paste0(out_dir, "/SAVERX_temp_test.mtx")
Matrix::writeMM(test.x, file = test_mtx_file)
pred_mtx_file = test_mtx_file
nonmissing_indicator = 1
tmp = api$autoencode(mtx_file = mtx_file,
                     curve_file_name = curve_file_name,
                     pred_mtx_file = test_mtx_file,
                     nonmissing_indicator = 1,                      
                     out_dir = out_dir,
                     batch_size = batch_size, 
                     write_output_to_tsv = write_output_to_tsv)
est.autoencoder = t(tmp[[1]]) #32738 x 2179
est.mu = rowMeans(train.x) #32738
est.const = est.mu %*% t(rep(1, length(test.idx)))
err1 = rowMeans(dpois(as.matrix(test.x), as.matrix(est.autoencoder), log=TRUE))
err2 = rowMeans(dpois(as.matrix(test.x), as.matrix(est.const), log=TRUE))
err.autoencoder <- err.autoencoder + err1  
err.const <- err.const + err2



k=3
i = ss[k]
train.idx = idx.perm[-(((i-1)*n.test + 1):(i * n.test))]
test.idx = idx.perm[((i-1)*n.test + 1):(i * n.test)]
test.x = x[, test.idx]
train.x = x[, train.idx]
gnames <- rownames(train.x)
cnames <- colnames(train.x)
train.x <- Matrix::Matrix(train.x, sparse = T)
mtx_file <- paste0(out_dir, "/SAVERX_temp.mtx")
Matrix::writeMM(train.x, file = mtx_file)
gnames = rownames(test.x)
cnames = colnames(test.x)
test.x = Matrix::Matrix(test.x, sparse = T)
test_mtx_file = paste0(out_dir, "/SAVERX_temp_test.mtx")
Matrix::writeMM(test.x, file = test_mtx_file)
pred_mtx_file = test_mtx_file
nonmissing_indicator = 1
tmp = api$autoencode(mtx_file = mtx_file,
                     curve_file_name = curve_file_name,
                     pred_mtx_file = test_mtx_file,
                     nonmissing_indicator = 1,                      
                     out_dir = out_dir,
                     batch_size = batch_size, 
                     write_output_to_tsv = write_output_to_tsv)
est.autoencoder = t(tmp[[1]]) #32738 x 2179
est.mu = rowMeans(train.x) #32738
est.const = est.mu %*% t(rep(1, length(test.idx)))
err1 = rowMeans(dpois(as.matrix(test.x), as.matrix(est.autoencoder), log=TRUE))
err2 = rowMeans(dpois(as.matrix(test.x), as.matrix(est.const), log=TRUE))
err.autoencoder <- err.autoencoder + err1  
err.const <- err.const + err2




## combined one final autoencoder
est.mu = rowMeans(x)
est.const = est.mu %*% t(rep(1, ncol(x)))
gnames <- rownames(x)
cnames <- colnames(x)

api <- python.module$api
print(api)

gnames <- rownames(x)
cnames <- colnames(x)
x <- Matrix::Matrix(x, sparse = T)
mtx_file <- paste0(out_dir, "/SAVERX_temp.mtx")
Matrix::writeMM(x, file = mtx_file)
rm(x)
gc()
test_mtx_file=NULL
tmp = api$autoencode(mtx_file = mtx_file,
                     curve_file_name = curve_file_name,
                     pred_mtx_file = test_mtx_file,
                     nonmissing_indicator = 1,                      
                     out_dir = out_dir,
                     batch_size = batch_size, 
                     write_output_to_tsv = write_output_to_tsv)

x.autoencoder = list()
x.autoencoder$result = t(tmp[[1]])
x.autoencoder$dispersion = tmp[[2]]
x.autoencoder$pi = t(tmp[[3]])
rm(tmp); gc()

x.autoencoder$result[err.autoencoder < err.const, ] <- est.mu[err.autoencoder < err.const]



