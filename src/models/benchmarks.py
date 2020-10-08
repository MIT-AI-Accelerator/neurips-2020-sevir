"""
Benchmarks
"""


# Optical flow model using rainy motion library
class OpticalFlow:
    def __init__(self,n_out=12):
        from rainymotion.models import Dense
        self.n_out=n_out
        self.norm={'scale':47.54,'shift':33.44} # TODO: don't hardcode this..
    def fit(self,X,y):
        # Doesn't need a fit
        pass
    def predict(self,X):
        """
        To make compatible with the pretrained CNNs, we assume data is scaled using self.norm.
        Data is unscaled to [0-255] before being passed to Dense(), and then rescaled back and returned.
        """
        y_pred = np.zeros((X.shape[0],X.shape[1],X.shape[2],self.n_out),dtype=np.float32)
        to_input = np.transpose(X, [0, 3, 1, 2]) # rainy motion expects [T, L, W]
        
        # Run optical flow on each sample
        model = Dense()
        # keep rainymotion defaults
        model.lead_steps = self.n_out
        model.of_method = "DIS"
        model.direction = "backward"
        model.advection = "constant-vector"
        model.interpolation = "idw"
        for x in range(to_input.shape[0]):
            model.input_data = to_input[x]*self.norm['scale']+self.norm['shift']
            to_output = model.run()
            y_pred[x] = np.transpose(to_output, [1, 2, 0]) # back to [L,W,T]
        return (y_pred-self.norm['shift'])/self.norm['scale']