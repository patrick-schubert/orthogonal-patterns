import tensorflow as tf

class Orthogonalizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, x):
    
        x = tf.reshape(x, (-1, x.shape[-1]))
        proj = tf.transpose(x) @ x
        lower_triangular = tf.linalg.LinearOperatorLowerTriangular(proj).to_dense()
        diag = proj * tf.eye(proj.shape[-1])
        loss = self.factor * tf.norm(lower_triangular - diag) #/ ( (x.shape[-1] * (x.shape[-1] - 1 )) / 2 )

        return loss

    def get_config(self):
        return {"factor": self.factor}
    
    def apply_reg(x):
        for layer in x.layers[:-1]:
            if hasattr(layer, 'kernel_regularizer') and isinstance(layer, tf.keras.layers.Dense):
                layer._handle_weight_regularization(layer.name, layer.kernel, Orthogonalizer(1.0))

class Regularizer_Sum(tf.keras.metrics.Metric):

    def __init__(self, name='reg', **kwargs):
        super(Regularizer_Sum, self).__init__(name=name, **kwargs)
        self.reg = self.add_weight(name='reg', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.reg.assign(sum(model.losses))


    def result(self):
        return self.reg