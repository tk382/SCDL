#' R interface for the autoencode python function
#'
#' @param x Target sparse data matrix of gene by cell. When pretraining is used, the genes should be the same as the nodes used in the pretrained model. If a node gene is missing is the target dataset, set all values of that gene as 0 in \code{x} and indicate that using \code{nonmissing_indicator}
#' @param python.module The python module for the Python package \code{sctransfer}
#' @param main A Python main module
#' @param nonmissing_indicator A single value 1 or a vector of 0 and 1s to indicate which nodes are missing in the target dataset. Set to 1 for no pretraining.
#' @param test.x Data matrix to evaluate the test error
#' @param  model.species Should be either 'Human' or 'Mouse' when pretraining is used
#' @param ... Extra parameters passed to Python module \code{sctransfer} function \code{api} (if no pretraining) or function \code{api_pretrain} (with pretraining).
#' @param write_output_to_tsv If True, then the result of Python is written as .tsv files instead of passing back to R. Default is False.
#'
#' @return A data matrix for the Python autoencoder result 
#'
#' @export
autoencode <- function(x, 
                       curve_file_name,
                       python.module,
                       main,
                       test.x = NULL,  
                       nonmissing_indicator = 1, 
                       out_dir = ".",
                       batch_size = 32L,
                       write_output_to_tsv = F,
                       ...) {
  

  api <- python.module$api
  print(api)
  
  gnames <- rownames(x)
  cnames <- colnames(x)
  x <- Matrix::Matrix(x, sparse = T)
  mtx_file <- paste0(out_dir, "/SAVERX_temp.mtx")
  Matrix::writeMM(x, file = mtx_file)
  rm(x)
  gc()

  if (!is.null(test.x)) {
    gnames = rownames(test.x)
    cnames = colnames(test.x)
    test.x = Matrix::Matrix(test.x, sparse = T)
    test_mtx_file = paste0(out_dir, "/SAVERX_temp_test.mtx")
    Matrix::writeMM(test.x, file = test_mtx_file)
    rm(test.x)
    gc()
  } else {
    test_mtx_file <- NULL
  }
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

  reticulate::py_run_string("
import gc
gc.collect()")
  return(x.autoencoder)
}
