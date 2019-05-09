import numpy as np
import tensorflow as tf
from keras import backend as K

def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x) + np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32)) #number of elements
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

PiAct = lambda x: tf.clip_by_value(x, 0, 0.95) 

class NB(object):
    def __init__(self, curve = None, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False, nonmissing_indicator=None):

        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta
        self.curve = curve

        if nonmissing_indicator is None:
            nonmissing_indicator = 1
        self.nonmissing_indicator = nonmissing_indicator

    def set_nonmissing(self, nonmissing_indicator):
        self.nonmissing_indicator = nonmissing_indicator

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)


            theta = tf.minimum(self.theta, 1e6)

            # (theta + y_true + eps) choose (theta + eps)             
            t1 = tf.lgamma(theta+eps) + tf.lgamma(y_true+1.0) - tf.lgamma(y_true+theta+eps)
            t2 = (theta+y_true) * tf.log(1.0+(y_pred/(theta+eps))) + (y_true * (tf.log(theta+eps) - tf.log(y_pred + eps)))

            final = t1 + t2

            if mean:
                if self.masking:
                    # final = tf.divide(tf.reduce_sum(final*non-nonmissing_indicator), nelem)
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    # final = tf.reduce_mean(final*nonmissing_indicator)
                    final = tf.reduce_mean(final)

        return final


class ZINB(NB):
    # ZINB is a subclass(?) of NB and there could be no extra work to be done.
    # *Given pi* this computes the loss. So loss computation remains the same
    # how do we incoroprate oen extra parameter?
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope = scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            nb_case = super().loss(y_true, y_pred, mean=False) -tf.log(1.0 - self.pi + eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta/(theta+y_pred+eps), theta)
            zero_case = -tf.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)

            if mean:
                nonmissing_indicator = self.nonmissing_indicator
                if self.masking:
                    result = reduce_mean(result*nonmissing_indicator)
                else:
                    result = tf.reduce_mean(result*nonmissing_indicator)

            result = _nan2inf(result)

        return result



class decayModel(NB):
    # ZINB is a subclass(?) of NB and there could be no extra work to be done.
    # *Given pi* this computes the loss. So loss computation remains the same
    # how do we incoroprate oen extra parameter?
    def __init__(self, curve, pi, scope='decay_loss/', **kwargs):
        super().__init__(scope = scope, **kwargs)
        self.pi = pi
        self.curve = curve
    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        curve = self.curve
        curve = tf.cast(curve, tf.float32)
        # self.pi = PiAct(curve[1] * K.exp(curve[0]-K.exp(curve[2]) * y_pred))
        with tf.name_scope(self.scope):
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.log(1.0 - self.pi + eps)
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta/(theta+y_pred+eps), theta)
            zero_case = -tf.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)

            if mean:
                nonmissing_indicator = self.nonmissing_indicator
                if self.masking:
                    result = reduce_mean(result*nonmissing_indicator)
                else:
                    result = tf.reduce_mean(result*nonmissing_indicator)

            result = _nan2inf(result)

        return result
