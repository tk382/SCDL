#' the SAVERX function
#'
#' Output and intemediate files are stored at the same directory as the input data file \code{text.file.name}. To avoid potential file name conflicts, please make the folder only contain \code{text.file.name}. DO NOT run two SAVER-X tasks for the data file in the same folder.
#'
#' @inheritParams computeShrinkage
#' @inheritParams computePrediction
#' @param verbose Whether to show more autoencoder optimization progress or not
#' @return the final denoised RDS data file saved in the same directory as the input data
#' @export
#' 
saverx_tae <- function(text.file.name, 
				   model.nodes.ID = NULL,
           is.large.data = F,
           verbose = F, batch_size = NULL, 
           clearup.python.session = T, ...) {

  computePrediction(text.file.name, 
                    model.nodes.ID, 
                    verbose = verbose,
                    is.large.data = is.large.data, 
                    batch_size = batch_size,
                    clearup.python.session = clearup.python.session, 
                    ...)

}
