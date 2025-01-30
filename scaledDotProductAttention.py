import tensorflow as tf

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, use_causal_mask=False):
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) 
    scaled_scores = scores / tf.math.sqrt(d_k) 
    if use_causal_mask:
        scaled_scores = mask_attn_weights(scaled_scores) 
    weights = tf.nn.softmax(scaled_scores, axis=-1)
    output = tf.matmul(weights, v)
    return output
# function test
train_source_embedded = tf.random.normal((2, 5, 16))

with tf.device('cpu:0'):
    input = train_source_embedded
    input = tf.expand_dims(input, axis=1)
    print("Scaled dot product attention (shape):", scaled_dot_product_attention(input, input, input, use_causal_mask=False).shape)
