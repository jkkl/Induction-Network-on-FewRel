import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[])
b = tf.constant(1, dtype=tf.int32)

tf.summary.scalar("a", a)
tf.summary.scalar("b", b)

sess = tf.Session()

init_op = tf.global_variables_initializer()
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter("train", sess.graph)

# sess.run(init_op)

for step in range(6):
    feed_dict = {a: step}
    summary = sess.run(merged_summaries, feed_dict=feed_dict)
    writer.add_summary(summary=summary, global_step=step)
