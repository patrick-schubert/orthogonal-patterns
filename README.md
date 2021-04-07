# Orthogonal Patterns in Tensorflow 2

Project from November - 2020

This repository contains code for the **Orthogonal Patterns in Tensorflow 2** Project.

### Requirements:

`Tensorflow` >= 2.3.0

### Example:

```python3
model = create_model()
model.compile(optimizer=tf.optimizers.Adam(),
                loss='categorical_crossentropy',  metrics=['accuracy', Regularizer_Sum()]) #If want to inspect Ortho Loss during training
Orthogonalizer.apply_reg(model)

```

Complete code with preliminary results can be found here: [Lab](https://colab.research.google.com/drive/1RbAchoIPilIzlGixRP4btuoC1HLR1l4B?usp=sharing)
