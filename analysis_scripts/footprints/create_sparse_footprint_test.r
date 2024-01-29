#setwd(dirname(getwd()))
source('/projects/0/ctdas/PARIS/transport_models/STILT_Model/stiltR/create_sparse_footprints_coords.r')

numpix.x=250
numpix.y=390

#emissgrid.all      <- matrix(sample(x = 0:1e-5, size = numpix.x * numpix.y, replace = TRUE),nrow=numpix.y,ncol=numpix.x) #initialize a fine grid matrix with randomly distributed values
emissgrid.all   <- matrix(0,nrow=numpix.y,ncol=numpix.x) #initialize a fine grid matrix with randomly distributed values
emissgrid.all[sample(seq_along(emissgrid.all), 5000, replace=FALSE)] <- 0.000001231251231246123
#emissgrid.all

lons = seq(-14.9, -14.9 + 250 * 0.2, 0.2)
lats = seq(33.05, 33.05 + 390 * 0.1, 0.1)

result <- create_sparse_matrix_coords(emissgrid.all, lons = lons, lats = lons)
print(class(result))
print(result$lon)
result$lon[1:50]
result$infl[1]



#result$index_x[1:50]
#result$index_y[1:50]