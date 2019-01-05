import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat

def visclassifier(fun,xTr,yTr,w=[],b=0):
	"""
	visualize decision boundary
	Define the symbols and colors we'll use in the plots later
	"""

	yTr = np.array(yTr).flatten()
	w = np.array(w).flatten()

	symbols = ["ko","kx"]
	marker_symbols = ['o', 'x']
	mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
	classvals = np.unique(yTr)

	plt.figure()

	res=300
	xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
	yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
	pixelX = repmat(xrange, res, 1)
	pixelY = repmat(yrange, res, 1).T

	xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

	testpreds = fun(xTe)
	Z = testpreds.reshape(res, res)
	# Z[0,0] = 1 # optional: scale the colors correctly
	plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

	for idx, c in enumerate(classvals):
		plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
           )

	if w != []:
		alpha = -1 * b / (w ** 2).sum()
		plt.quiver(w[0] * alpha, w[1] * alpha,
			w[0], w[1], linewidth=2, color=[0,1,0])

	plt.axis('tight')
	plt.show()