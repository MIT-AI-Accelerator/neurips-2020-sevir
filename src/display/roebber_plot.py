"""
Roebber plot.   Code based on https://johnrobertlawson.github.io/evac/_modules/evac/plot/performance.html
"""
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt

class RoebberPlot(object):
    
    def __init__(self,ax):
        # Plotting settings
        self.ax=ax
        self.contours = {}
        self.contours['bias'] = N.array([0.25,0.5,1.0,2.0,4.0])
        self.contours['csi'] = N.arange(0.1,1.1,0.1)
        # self.xticks = 201
        self.xticks = 401
        self.yticks = self.xticks
        self.sr_x = N.linspace(0,1,self.xticks)
        self.pod_y = N.linspace(0,1,self.yticks)

        self.create_axis()
        self.plot_bias_lines()
        self.plot_csi_lines()

    def create_axis(self):
        self.ax.grid(False)

        self.ax.set_xlim([0,1])
        xlab = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
        self.ax.set_xticks(xlab)
        self.ax.set_xticklabels(xlab)
        self.ax.set_xlabel("Success Ratio")

        self.ax.set_ylim([0,1])
        ylab = xlab
        self.ax.set_yticks(ylab)
        self.ax.set_yticklabels(ylab)
        self.ax.set_ylabel("Probability of Detection")

        self.ax.set_aspect('equal')


    def plot_bias_lines(self):
        bias_arr = self.compute_bias_lines()
        col = [.2,.2,.2]
        for bn, b in enumerate(self.contours['bias']):
            # self.ax.plot(bias_arr[:,0,bn],bias_arr[:,1,bn],
            self.ax.plot(bias_arr[:,1,bn],self.pod_y,
                            color=col,lw=1,linestyle='--')
            bstr = '{:1.1f}'.format(b)
            # pdb.set_trace()
            if b < 1:
                xpos = 1.00
                ypos = b
            else:
                xpos = 1.00/b
                ypos = 1.00
            self.ax.annotate(bstr,xy=(xpos,ypos),xycoords='data',
                            # bbox=dict(fc='red'),color='white',
                            fontsize=7,color=col,
                            xytext=(2,3),textcoords='offset points',
                            )


    def plot_csi_lines(self):
        xpos = 0.945
        csi_arr = self.compute_csi_lines()
        # mid = N.floor(csi_arr.shape[0]/2)
        # mid = N.int(csi_arr.shape[0] - (0.975*csi_arr.shape[0]))
        nc = len(self.contours['csi'])
        col = [.2,.2,.2]
        for cn, c in enumerate(self.contours['csi']):
            # csi_c = self.ax.plot(csi_arr[:,0,cn],csi_arr[:,1,cn],
            self.ax.plot(self.sr_x,csi_arr[:,1,cn],
                                color=col,lw=1,)
            cstr = '{:1.1f}'.format(c)
            # self.ax.clabel(csi_c[1],fontsize=8,inline=True,fmt='%1.1f')
            # pdb.set_trace()
            # if not N.isnan(csi_arr[mid,1,cn]):
            if True:
                # negx = int(negx_factor*self.xticks)
                # yidx = N.abs(csi_arr[:,0,cn]-(1-negx_factor)).argmin()
                ypos = 1/((1/c) + 1 - (1/xpos))
                # self.fig.text(N.ones(nc)*0.93,csi_arr[mid,1,cn],
                                # cstr,color='blue',)#facecolor='white')
                self.ax.annotate(cstr,xy=(xpos-0.01,ypos),
                # self.ax.annotate(cstr,xy=(1-negx_factor,csi_arr[mid,1,cn]),
                                    xycoords='data', fontsize=7, 
                                    #color='white', bbox=dict(fc='blue'),
                                    bbox=dict(fc='white',color='white',pad=0),
                                    xytext=(3,3),textcoords='offset points',
                                    color=col)

        # pdb.set_trace()


    def compute_bias_lines(self):
        bias_arr = N.zeros([self.sr_x.size,2,self.contours['bias'].size])
        for bn, b in enumerate(self.contours['bias']):
            bias_arr[:,0,bn] = self.pod_y/b
            bias_arr[:,1,bn] = self.sr_x/b
        bias_arr[bias_arr>1.0] = 1.0
        # pdb.set_trace()
        return bias_arr


    def compute_csi_lines(self):
        # might need to switch around the axes
        csi_arr = N.zeros([self.sr_x.size,2,self.contours['csi'].size])
        for cn, c in enumerate(self.contours['csi']):
            # x coordinates for this CSI contour
            sr_x_c = 1/((1/c) + 1 - (1/self.pod_y))
            # and y coordinates
            pod_y_c = 1/((1/c) + 1 - (1/self.sr_x))

            # assign to array
            csi_arr[:,0,cn] = sr_x_c
            csi_arr[:,1,cn] = pod_y_c

        # pdb.set_trace()
        csi_arr[csi_arr < 0] = N.nan
        csi_arr[csi_arr > 1] = N.nan
        return csi_arr