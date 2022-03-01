'''
This module contains utility functions for Lecture 12.
'''


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap





def average_distance_from_mean(y):
	'''
	Compute the root-mean-squared error (RMSE) for a set of observations.

	The RMSE represents the average Euclidian distance from the mean.
	
	**Inputs**:
	
	* y : (m,) or (m,n) data array

	**Outputs**:
	
	* e 
	
	**y** : (m,) or (m,n) data array

	'''
	y   = np.array(y)         # ensure that the data is an array
	d   = y - y.mean(axis=0)  # vectors between observations and mean
	if y.ndim > 1:           # if more than one variable
		d = np.linalg.norm( d, axis=1 )  # compute Euclidian norms (i.e., vector lengths)
	avg = np.abs(d).mean()    # average distances
	# avg = (d**2).mean()    # average squared distances
	return avg




def average_distances(x, labels, ratios=False):
	'''
	Compute average within- and between-cluster distances.

	**Inputs**:

	* x : (m,n) data array  (m: number of observations,  n: number of variables)

	* labels : (m,) array of integer cluster labels

	* ratios : True or False  (if True, MSE values will be normalized by between-cluster distance)

	**Outputs**:

	* dw : array of within-cluster average distances values (one per cluster)
	
	* db : average between-cluster distance
	'''
	f       = average_distance_from_mean
	ulabels = np.unique(labels)  # unique labels
	dw      = [f( x[labels==i] )  for i in ulabels]  # within-cluster mse values (one per cluster)
	means   = np.array( [x[labels==i].mean(axis=0)  for i in ulabels] )  # cluster mean locations
	db      = f(means)  # between-cluster distances
	if ratios:
		dw /= db
		db  = 1.0
	return dw,db
	
	
	

def classification_rate(labels, labels_predicted):
	y  = np.asarray(labels)
	yp = np.asarray(labels_predicted)
	return (y==yp).mean()
	



def get_random_train_test_sets(x, y, ntest=1):
	x            = np.asarray(x)
	y            = np.asarray(y)
	n            = y.size
	# boolean indices:
	ind          = np.random.randint(0, n, ntest)
	i_train      = np.array( [True]*n )
	i_train[ind] = False
	i_test       = np.logical_not(i_train)
	# divide sets:
	x_train      = x[i_train]
	x_test       = x[i_test]
	y_train      = y[i_train]
	y_test       = y[i_test]
	return x_train, x_test, y_train, y_test










def plot_decision_surface(knn, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.7, ax=None):
	x = knn._fit_X
	y = knn._y

	xmin,xmax = x.min(axis=0), x.max(axis=0)

	X0p     = np.linspace( xmin[0] , xmax[0] , n )
	X1p     = np.linspace( xmin[1] , xmax[1] , n )
	X0p,X1p = np.meshgrid( X0p , X1p )
	xp      = np.vstack( [X0p.flatten(), X1p.flatten()] ).T
	yp      = knn.predict(xp)

	Yp      = np.reshape(yp, X0p.shape)


	cmap     = ListedColormap(colors)
	
	
	ax     = plt.gca() if (ax is None) else ax
	
	for yy,cc in zip(np.unique(y), colors):
		xx = x[y==yy]
		ax.scatter( xx[:,0], xx[:,1], color=cc, s=marker_size, alpha=marker_alpha, label=f'Label = {yy}' )
	ax.pcolormesh(X0p, X1p, Yp, cmap=cmap, alpha=alpha)
	ax.axis('equal')
	ax.legend()




def plot_labeled_points(x, labels, colors=None, ax=None, ms=12):
	x      = np.asarray(x)
	assert x.ndim < 3, 'Only one- or two-dimensional arrays supported.'
	if x.ndim == 2:
		assert x.shape[1]<3, 'Too many variables (n=%d). n must be 1 or 2.' %x.shape[1]
	ax     = plt.gca() if (ax is None) else ax
	uy     = np.unique(labels)
	for yy,cc in zip(uy,colors):
		xx = x[labels==yy]
		if x.ndim==1:
			ax.plot( xx, np.zeros(xx.size), 'o', color=cc, ms=ms, label=f'Label = {yy}' )
		elif x.ndim==2:
			ax.plot( xx[:,0], xx[:,1], 'o', color=cc, ms=ms, label=f'Label = {yy}' )
	ax.axhline(0, color='k', ls=':')
	ax.legend()


