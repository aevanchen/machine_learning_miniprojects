import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

def loaddata(filename):
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T

def visualize_knn_2D(findknn): 
    global N,k,hp,x,hl,xt,ht
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(0,1)
    plt.ylim(0,1)
    N=50;
    k=5;
    x=np.random.rand(N,2)
    hp=plt.plot(x[:,0],x[:,1],'bo')
    xt=None
    hl=None
    ht=None


    def onclick(event):
        global N,k,hp,x,hl,xt,ht

        if ht is None:        
            ht=plt.plot(event.xdata,event.ydata,'ro')    
        ht[0].set_data(event.xdata,event.ydata)
        xt=np.array([ht[0].get_data()])
        inds,dists=findknn(x,xt,k); # find k nearest neighbors 
        xdata=[]
        ydata=[]

        for i in range(k):
                xdata.append(xt[0,0])
                xdata.append(x[inds[i,0],0])
                xdata.append(None)
                ydata.append(xt[0,1])
                ydata.append(x[inds[i,0],1])
                ydata.append(None)
        if hl is None:
            hl=plt.plot(xdata,ydata,'r-')
            plt.title('%i-Nearest Neighbors' % k)
        else:
            hl[0].set_data(xdata,ydata)

        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('Click to add test point')
    
def visualize_knn_images(findknn, imageType='faces'):
    global who, xTr, xTe, yTr, yTe, xdim, ydim
    fig = plt.figure()
    who=0
    
    if imageType == 'faces':
        xTr,yTr,xTe,yTe = loaddata("faces.mat")
        xdim = 38
        ydim = 31 
    else:
        xTr,yTr,xTe,yTe = loaddata("digits.mat")
        xdim = 16
        ydim = 16 

    # normalize all pixel values to [0,1]
    ma = np.max(xTe.flatten())
    mi = np.min(xTe.flatten())
    xTe = (xTe-mi)/(ma-mi)
    ma = np.max(xTr.flatten())
    mi = np.min(xTr.flatten())
    xTr = (xTr-mi)/(ma-mi)

    def onclick(event):
        global who, xTr, xTe, yTr, yTe, xdim, ydim
        sb=1
        for i in range(4):
            plt.subplot(4,4,sb)
            plotimage(xdim, ydim, xTe[who, :].reshape(ydim, xdim).T,2)
            plt.axis('off')
            sb+=1

            indices, dists = findknn(xTr,np.array(xTe[who,:], ndmin=2), 3)
            for j in range(3):            
                plt.subplot(4,4,sb)
                plotimage(xdim, ydim, xTr[indices[j], :].reshape(ydim, xdim).T,int(yTr[indices[j]]==yTe[who]))
                plt.axis('off')
                sb+=1
            who+=1
        plt.subplot(4,4,1);plt.title('TEST')
        plt.subplot(4,4,2);plt.title('1-NN')
        plt.subplot(4,4,3);plt.title('2-NN')
        plt.subplot(4,4,4);plt.title('3-NN')

    onclick(None)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print('Click on the images above, to cycle through the test images.')

def plotimage(xdim, ydim, M,d=2): # plot an image and draw a red/blue/green box around it (specified by "d")
    Q=np.zeros((xdim+2,ydim+2))
    Q[1:-1,1:-1]=M
    Q=np.repeat(Q[:,:,np.newaxis],3,axis=2)
    Q[[0,-1],:,d]=1
    Q[:,[0,-1],d]=1
    plt.imshow(Q, cmap=plt.cm.binary_r)
    
def plotfaces(X, xdim=38, ydim=31 ):
    n, d = X.shape            
    m=np.ceil(np.sqrt(n))
    for i in range(n):
        plt.subplot(m,m,i+1)
        plt.imshow(X[i, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
        plt.axis('off')
        
def visualize_knn_boundary(knnclassifier):
    globalK=np.ones(1, dtype=int);
    Xdata=[]
    ldata=[]
    w=[]
    b=[]
    line=None
    stepsize=1;

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(0,1)
    plt.ylim(0,1)

    def visboundary(k, Xdata, ldata, ax, knnclassifier):
        RES=50;
        grid = np.linspace(0,1,RES);
        X, Y = np.meshgrid(grid, grid)
        xTe=np.array([X.flatten(),Y.flatten()])
        Z=knnclassifier(np.array(Xdata),np.array(ldata).T,xTe.T,k[0])
        Z=Z.reshape(X.shape)
        boundary=ax.contourf(X, Y, np.sign(Z),colors=[[0.5, 0.5, 1], [1, 0.5, 0.5]])
        plt.draw()
        sys.__stdout__.write(str(Xdata))
        return(boundary)

    def onclickkdemo(event, Xdata, ldata, globalK, ax, knnclassifier):
        if event.key == 'p': # add positive point
            ax.plot(event.xdata,event.ydata,'or')
            label=1
            pos=np.array([event.xdata,event.ydata])
            ldata.append(label);
            Xdata.append(pos)
        if event.key==None: # add negative point
            ax.plot(event.xdata,event.ydata,'ob')
            label=-1
            pos=np.array([event.xdata,event.ydata])
            ldata.append(label);
            Xdata.append(pos)
        if event.key == 'h':
            globalK=(globalK+1) % len(ldata)+1;
        visboundary(globalK, Xdata, ldata, ax, knnclassifier)
        plt.title('k=%i' % globalK)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclickkdemo(event, Xdata, ldata, globalK, ax, knnclassifier))
