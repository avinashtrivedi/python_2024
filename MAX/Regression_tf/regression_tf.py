import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt

# Generate synthetic data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))

# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Plot the data
plt.scatter(x_zeros[:,0], x_zeros[:,1], c='r', marker='x', label='y=0')
plt.scatter(x_ones[:,0], x_ones[:,1], c='b', marker='x', label='y=1')
plt.legend()
plt.title("Synthetic Data")
plt.show()

# tf.compat.v1.disable_eager_execution()
# import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

# Generate a TensorFlow graph
with tf.name_scope("placeholders"):
  x = tf.compat.v1.placeholder(tf.float32, (N, 2))
  y = tf.compat.v1.placeholder(tf.float32, (N,))

with tf.name_scope("weights"):
  W = tf.Variable(tf.random.normal((2, 1)))
  b = tf.Variable(tf.random.normal((1,)))

with tf.name_scope("prediction"):
  y_logit = tf.squeeze(tf.matmul(x, W) + b)
  # the sigmoid gives the class probability of 1
  y_one_prob = tf.sigmoid(y_logit)
  # Rounding P(y=1) will give the correct prediction.
  y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
  # Compute the cross-entropy term for each datapoint
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
  # Sum all contributions
  l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
  train_op = tf.compat.v1.train.AdamOptimizer(.01).minimize(l)

with tf.name_scope("summaries"):
  tf.compat.v1.summary.scalar("loss", l)
  merged = tf.compat.v1.summary.merge_all()
  train_writer = tf.compat.v1.summary.FileWriter('logistic-train', tf.compat.v1.get_default_graph())

# Train the model, get the weights, and make predictions
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(20000):
        _, summary, loss = sess.run([train_op, merged, l], {x: x_np, y: y_np})

        if i % 1000 == 0:
            print("step %d, loss: %f" % (i, loss))

        train_writer.add_summary(summary, i)

    W_np, b_np, y_pred_np = sess.run([W, b, y_pred], {x: x_np})

#Plot the predicted outputs on top of the data:
plt.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_np, cmap="coolwarm")
plt.scatter(x_zeros[:,0], x_zeros[:,1], c='r', marker='x', label='y=0')
plt.scatter(x_ones[:,0], x_ones[:,1], c='b', marker='x', label='y=1')
plt.legend()
plt.show()



