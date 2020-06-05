"""
Various loss functions for training 

"""
import os
import tensorflow as tf

# Global vars for VGG19 loss
global vggfull
global vggfeats
global vggfeats_l
global vggfeats_seq
global vggfeats_seq_l
global vgginput_shape
global vgginit

# Download this file using keras first.
_default_vgg_weights = f'{os.environ["HOME"]}/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VGGLoss():
    def __init__(self,
                 input_shape=(None,192,192,1),
                 resize_to=None,
                 layer='block5_conv4',
                 vgg_weights=_default_vgg_weights,
                 normalization_scale=1.0,
                 normalization_shift=0.0,
                 skip=1):
        if len(input_shape)==4:
            self.T,self.L,self.W,self.D = input_shape
            self.use_seq=True
        elif len(input_shape)==3:
            self.L,self.W,self.D = input_shape
            self.use_seq=False
        else:
            raise ValueError('input_shape must have length 3 or 4')

        assert(self.D==1 or self.D==3) # Depth channel must be gray scale or RGB
        self.skip=skip
        self.resize_to=resize_to
        if resize_to:
            self.vggshape = (resize_to[0],resize_to[1],3) 
        else:
            self.vggshape = (self.L,self.W,3) 
        self.vggfull = tf.keras.applications.VGG19(weights=vgg_weights,
                                                   input_shape=self.vggshape,
                                                   include_top=False)
        self.vggfull.trainable=False
        for l in self.vggfull.layers:
            l.trainable=False
            
        # Build a pipeline that computes the MSE between VGG features.
        # Becuase VGG featueres are big, we will perform the computation sequentually over
        # the time dimension.
        self.vggfeats = tf.keras.Model(self.vggfull.input, 
                                  outputs=self.vggfull.get_layer(layer).output)
        
        if self.use_seq:
            # Make the sequence-ified version of self.vggfeats
            # tf.keras.layers.TimeDistributed only accepts single inputs, so 
            # will assume inputs are concat-ed along channel dim
            inp = tf.keras.layers.Input(shape=(self.vggshape[0],self.vggshape[1],6)) # inputs concatted along D
            inp_norm = tf.keras.layers.Lambda(lambda inp,scale,shift: inp*scale+shift,
                                                arguments={'scale':np.float32(normalization_scale),
                                                           'shift':np.float32(normalization_shift)})(inp)
            yt = tf.keras.layers.Lambda( lambda inp: inp[:,:,:,:3] )(inp_norm)
            yp = tf.keras.layers.Lambda( lambda inp: inp[:,:,:,3:] )(inp_norm)
            f_true = self.vggfeats(yt)
            f_pred = self.vggfeats(yp)
            f_diff = tf.keras.layers.Lambda( lambda f: tf.math.square(f[0]-f[1]))([f_true,f_pred])
            out    = tf.keras.layers.Lambda( lambda d: tf.reduce_mean(d,axis=[1,2,3]) )(f_diff)
            self.vggfeats_model = tf.keras.Model(inputs=inp,outputs=out,name="VGG_Model")
            
            # Build TimeDistributed that applys this over a sequence
            inp_seq           = tf.keras.layers.Input(shape=(self.T,self.vggshape[0],self.vggshape[1],6))
            vggfeats_td       = tf.keras.layers.TimeDistributed(self.vggfeats_model,name='VGG_Sequence')(inp_seq)
            self.vggfeats_seq = tf.keras.Model(inputs=inp_seq, outputs=vggfeats_td,name='VGG_Sequence_Loss')

        #####
        # Pipeline to prep data prior to being passed to vgg functions
        #####
        loss_inp = tf.keras.layers.Input(shape=(self.L,self.W,1)) 
        if self.resize_to:
            loss_inp_resized = tf.keras.layers.Lambda(
                lambda t,size:  tf.image.resize(t,size),
                arguments={'size':(self.vggshape[0],self.vggshape[1])} )(loss_inp)
        else:
            loss_inp_resized=loss_inp
        
        if self.D==1:
            loss_inp_tile3 = tf.keras.layers.Lambda(lambda t: tf.tile(t,[1,1,1,3]))(loss_inp_resized)
        else:
            loss_inp_tile3 = loss_inp_resized
        self.prep_loss_input = tf.keras.Model(inputs=loss_inp,outputs=loss_inp_tile3,name='PrepLossInputs')
        
        if self.use_seq:
            # Create sequence-ified version of self.prep_loss_input
            loss_inp_seq = tf.keras.layers.Input(shape=(self.T,self.L,self.W,1))
            if self.skip>1:
                # subset the time sequence
                loss_inp_seq_skipped = tf.keras.layers.Lambda(
                          lambda t,skip: t[:,::skip],
                          arguments={'skip':self.skip})(loss_inp_seq)
            else:
                loss_inp_seq_skipped = loss_inp_seq
            prep_loss_input_td = tf.keras.layers.TimeDistributed(self.prep_loss_input,name='PrepLossInputsSequence')(loss_inp_seq_skipped)
            self.prep_loss_input_seq = tf.keras.Model(inputs=loss_inp_seq,outputs=prep_loss_input_td)



    def get_loss(self):
        def vggloss(y_true,y_pred):
            """
            If use_seq, inputs should be [N,T,L,W,D]
            If not,     inputs should be [N,L,W,D]
            """
            if not self.use_seq:
                yt_inp = tf.keras.layers.Input(shape=(self.L,self.W,self.D)) 
                yp_inp = tf.keras.layers.Input(shape=(self.L,self.W,self.D))
                yt_inp_prepped=self.prep_loss_input(yt_inp) # resizes & tiles
                yp_inp_prepped=self.prep_loss_input(yp_inp) # resizes & tiles
                f_true = self.vggfeats(yt_inp_prepped)
                f_pred = self.vggfeats(yp_inp_prepped)
                f_diff = tf.keras.layers.Lambda( lambda f: tf.math.square(f[0]-f[1]))([f_true,f_pred])
                out    = tf.keras.layers.Lambda( lambda d: tf.reduce_mean(d,axis=[1,2,3]) )(f_diff)
                loss = tf.keras.Model(inputs=[yt_inp,yp_inp],outputs=out)
                d = loss([y_true,y_pred])
                return tf.reduce_mean(d)
            else:
                yt_inp = tf.keras.layers.Input(shape=(self.T,self.L,self.W,self.D)) 
                yp_inp = tf.keras.layers.Input(shape=(self.T,self.L,self.W,self.D))
                yt_inp_prepped = self.prep_loss_input_seq(yt_inp) # skips,resizes & tiles
                yp_inp_prepped = self.prep_loss_input_seq(yp_inp)
                # Must pass a single input to TimeDistributed, so concat inputs here
                y_concat = tf.keras.layers.Lambda(lambda ys: tf.concat((ys[0],ys[1]) ,axis=-1))([yt_inp_prepped,yp_inp_prepped])
                vgg=self.vggfeats_seq(y_concat)
                loss = tf.keras.Model(inputs=[yt_inp,yp_inp],outputs=vgg)
                d = loss([y_true,y_pred])
                return tf.reduce_mean( d ) # mean MSE over time
        return vggloss










def init_vgg19(input_shape=(192,192,1),
               T=None,
               layer='block5_conv4',
               vgg_weights=None,
               normalization_scale=1.0,
               normalization_shift=0.0):
    """
    Initializes the VGG loss pipeline.
    
    input_shape        Input shape tuple (L,W,D).  Inputs are resized to this shape before being passed to VGG    
    layer   str        name of VGG 19 layer to use for features
    vgg_weights        weights for VGG19
    T                  Size of T dimension
    normalization_scale   Scale for normalizing data before passing to VGG (normed_data = data*scale + shift)  
    normalization_shift   Shift for normalizing data before passing to VGG (normed_data = data*scale + shift) 

    
    Make sure weights are downloaded first to vgg_weights.
    """
    global vggfull
    global vggfeats
    global vggfeats_seq
    global vgginput_shape
    global vgginit
    
    L,W,D = input_shape
    vgginput_shape=input_shape
    assert(D==1 or D==3) # Depth channel must be gray scale or RGB
    if vgg_weights is None:
        vgg_weights = default_vgg_weights
    vggshape = (input_shape[0],input_shape[1],3) 
    vggfull = tf.keras.applications.VGG19(weights=vgg_weights,input_shape=vggshape,include_top=False)
    vggfull.trainable=False
    for l in vggfull.layers:
        l.trainable=False
        
    # Build a pipeline that computes the MSE between VGG features.
    # Becuase VGG featueres are big, we will perform the computation sequentually over
    # the time dimension.
    vggfeats = tf.keras.Model(vggfull.input, outputs=vggfull.get_layer(layer).output)
    
    # tf.keras.layers.TimeDistributed only accepts single inputs, so 
    # will assume inputs are concat-ed along channel dim
    inp = tf.keras.layers.Input(shape=(L,W,6)) # inputs concatted along D
    inp_norm = tf.keras.layers.Lambda(lambda inp: inp*normalization_scale+normalization_shift)(inp)
    yt = tf.keras.layers.Lambda( lambda inp: inp[:,:,:,:3] )(inp_norm)
    yp = tf.keras.layers.Lambda( lambda inp: inp[:,:,:,3:] )(inp_norm)
    f_true = vggfeats(yt)
    f_pred = vggfeats(yp)
    f_diff = tf.keras.layers.Lambda( lambda f: tf.math.square(f[0]-f[1]))([f_true,f_pred])
    out    = tf.keras.layers.Lambda( lambda d: tf.reduce_mean(d,axis=[1,2,3]) )(f_diff)
    vggfeats_model = tf.keras.Model(inputs=inp,outputs=out)
    
    inp_seq   = tf.keras.layers.Input(shape=(T,L,W,6))
    
    # Build TimeDistributed that applys this over a sequence
    vggfeats_td  = tf.keras.layers.TimeDistributed(vggfeats_model,name='VGG_Sequence')(inp_seq)
    vggfeats_seq = tf.keras.Model(inputs=inp_seq, outputs=vggfeats_td,name='VGG_Sequence_Loss')
    vgginit=True


    
def vgg19_loss(y_true, y_pred):
    """
    Input:
      y_true  N,T,L,L,D  target
      Y_pred  N,T,L,L,D  prediction
      
    Depth D must be either 3 or 1.  If equal to 1, channel is repeated 3 times to simulate RGB.
    
    Outputs MSE between Vgg19 features computed using layer.
    
    """
    global vggfull
    global vggfeats
    global vggfeats_seq
    global vgginit
    global vgginput_shape
    if not vgginit:
        raise ValueError('Please call init_vgg19 first')
        
    L,W,D = vgginput_shape
    #yt_inp = tf.keras.layers.Input(shape=(None,L,W,1)) 
    #yp_inp = tf.keras.layers.Input(shape=(None,L,W,1))
    
    T=y_true.get_shape()[1]
    Lin = y_true.get_shape()[2]
    Win = y_true.get_shape()[3]
    yt_inp = tf.keras.layers.Input(shape=(None,None,None,1)) 
    yp_inp = tf.keras.layers.Input(shape=(None,None,None,1))  

    yt_reshape = tf.keras.layers.Lambda(lambda t:  tf.reshape(t,[-1,384,384,1]) )(yt_inp)
    yp_reshape = tf.keras.layers.Lambda(lambda t:  tf.reshape(t,[-1,384,384,1]) )(yp_inp)

    yt_resize = tf.keras.layers.Lambda(lambda t:  tf.image.resize(t,(L,W)) )(yt_reshape)
    yp_resize = tf.keras.layers.Lambda(lambda t:  tf.image.resize(t,(L,W)) )(yp_reshape)
    
    yt_reshape2 = tf.keras.layers.Lambda(lambda t:  tf.reshape(t,[-1,T,L,W,D]) )(yt_resize)
    yp_reshape2 = tf.keras.layers.Lambda(lambda t:  tf.reshape(t,[-1,T,L,W,D]) )(yp_resize)

    yt_tile3 = tf.keras.layers.Lambda(lambda t: tf.tile(t,[1,1,1,1,3]))(yt_reshape2)
    yp_tile3 = tf.keras.layers.Lambda(lambda t: tf.tile(t,[1,1,1,1,3]))(yp_reshape2)
    y_concat = tf.keras.layers.Lambda(lambda ys: tf.concat((ys[0],ys[1]) ,axis=4))([yt_tile3,yp_tile3])
    y_concat = tf.keras.layers.Lambda(lambda y: y[:,::6])(y_concat) # every hour
    vgg = vggfeats_seq(y_concat)
    loss = tf.keras.Model(inputs=[yt_inp,yp_inp],outputs=vgg)
    d = loss([y_true,y_pred])
    return tf.reduce_mean( d ) # mean MSE over time
        
        
def recon_loss(img,decoded_img, decoded_str):
    """
    recon loss for VAEs
    """
    if int(tf.__version__[0])<2: # TF 1
        reconstruction_loss = -tf.reduce_sum( tf.contrib.distributions.Normal(
                                decoded_img,decoded_str ).log_prob(img), axis=[1,2,3],name='reconloss' )
    else:  # TF 2
        import tensorflow_probability as tfp
        reconstruction_loss = -tf.reduce_sum( tfp.distributions.Normal(
                      decoded_img, decoded_str).log_prob(img), axis=[1,2,3],name='reconloss' )
    return tf.reduce_mean(reconstruction_loss,axis=0)
       

        
def kl_loss(img,decoded_img,encoder_log_var,encoder_mu):
    """
    LK loss for VAEs
    """
    kl_loss = -0.5 * tf.reduce_sum( (1+encoder_log_var-tf.exp(encoder_log_var)-encoder_mu**2), axis=[1,2,3],name='klloss' )
    return tf.reduce_mean(kl_loss,axis=0)
        
     

