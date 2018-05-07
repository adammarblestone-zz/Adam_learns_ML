import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''See: https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # color maps
from sklearn.manifold import TSNE
tf.logging.set_verbosity(tf.logging.INFO)

# Check for GPU
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

devices = get_available_devices()

print(devices)

def MNIST_row_as_image(row):
    arr1 = []
    for i in range(28):
        arr2 = []
        for j in range(28):
            arr2.append(row[28*i + j])
        arr1.append(arr2)
    return np.array(arr1)
    
def tf_get_filter_banks(center_coords_dec, sigma_dec, stride_dec, N): # NxN grid of filters
    A,B = 28,28 # For MNIST
    sigma_dec = tf.exp(sigma_dec) # Note: sigma_decoded is the variance not the standard deviation here
    stride_dec = tf.exp(stride_dec) # These start as logs
    stride = stride_dec*float(A-1)/(N-1)
    cx = (center_coords_dec[0]+1)*(float(A+1)/2.0)
    cy = (center_coords_dec[1]+1)*(float(B+1)/2.0)
    center_coords = [cx, cy]

    means_of_filtersx = []
    means_of_filtersy = []
    normx = []
    normy = []
    epsilon = 0.0000001
    for i in range(N):
        x_mean = center_coords[0] + (i-N/2-0.5)*stride
        y_mean = center_coords[1] + (i-N/2-0.5)*stride
        means_of_filtersx.append(x_mean)
        means_of_filtersy.append(y_mean)
        normx.append(tf.maximum(tf.reduce_sum([tf.exp(-(ahat-means_of_filtersx[i])**2/(2*(sigma_dec))) for ahat in range(A)]), epsilon))
        normy.append(tf.maximum(tf.reduce_sum([tf.exp(-(bhat-means_of_filtersy[i])**2/(2*(sigma_dec))) for bhat in range(B)]), epsilon))

    filter_bank_X = [[(tf.exp(-(a-means_of_filtersx[i])**2/(2*(sigma_dec))))/normx[i] for a in range(A)] for i in range(N)]
    filter_bank_Y = [[(tf.exp(-(b-means_of_filtersy[i])**2/(2*(sigma_dec))))/normy[i] for b in range(B)] for i in range(N)]

    return filter_bank_X, filter_bank_Y

def np_get_filter_banks(center_coords_dec, sigma_dec, stride_dec, N): # NxN grid of filters
    A,B = 28,28 # For MNIST
    sigma_dec = np.exp(sigma_dec) # Note: sigma_decoded is the variance not the standard deviation here
    stride_dec = np.exp(stride_dec) # These start as logs
    stride = stride_dec*(float(A-1))/(N-1) 
    cx = (center_coords_dec[0]+1)*(float(A+1)/2.0)
    cy = (center_coords_dec[1]+1)*(float(B+1)/2.0)
    center_coords = [cx, cy]

    means_of_filtersx = []
    means_of_filtersy = []
    normx = []
    normy = []
    epsilon = 0.00000001
    for i in range(N):
        x_mean = center_coords[0] + (i-N/2-0.5)*stride
        y_mean = center_coords[1] + (i-N/2-0.5)*stride
        means_of_filtersx.append(x_mean)
        means_of_filtersy.append(y_mean)
        normx.append(np.max([np.sum([np.exp(-(ahat-means_of_filtersx[i])**2/(2*(sigma_dec))) for ahat in range(A)]), epsilon]))
        normy.append(np.max([np.sum([np.exp(-(bhat-means_of_filtersy[i])**2/(2*(sigma_dec))) for bhat in range(B)]), epsilon]))

    filter_bank_X = [[(np.exp(-(a-means_of_filtersx[i])**2/(2*(sigma_dec))))/normx[i] for a in range(A)] for i in range(N)]
    filter_bank_Y = [[(np.exp(-(b-means_of_filtersy[i])**2/(2*(sigma_dec))))/normy[i] for b in range(B)] for i in range(N)]

    return filter_bank_X, filter_bank_Y, [[x,y] for x in means_of_filtersx for y in means_of_filtersy]


def test_placement():
    # Test the placement
    img = mnist.train.images[10003][:]
    plt.figure()
    plt.imshow(MNIST_row_as_image(img), cmap = cm.Greys_r)
    _, _, means = np_get_filter_banks(center_coords_dec = [0,0], sigma_dec = 0, stride_dec = -3, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'r')
    _, _, means = np_get_filter_banks(center_coords_dec = [0,0], sigma_dec = 0, stride_dec = -2, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'b')
    _, _, means = np_get_filter_banks(center_coords_dec = [0,0], sigma_dec = 0, stride_dec = -1, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'g')
    _, _, means = np_get_filter_banks(center_coords_dec = [0,0], sigma_dec = 0, stride_dec = -0, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'y')
    plt.figure()
    plt.imshow(MNIST_row_as_image(img), cmap = cm.Greys_r)
    _, _, means = np_get_filter_banks(center_coords_dec = [-1,0], sigma_dec = 0, stride_dec = -3, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'r')
    _, _, means = np_get_filter_banks(center_coords_dec = [-1,0], sigma_dec = 0, stride_dec = -2, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'b')
    _, _, means = np_get_filter_banks(center_coords_dec = [-1,0], sigma_dec = 0, stride_dec = -1, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'g')
    _, _, means = np_get_filter_banks(center_coords_dec = [-1,0], sigma_dec = 0, stride_dec = -0, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'y')
    plt.figure()
    plt.imshow(MNIST_row_as_image(img), cmap = cm.Greys_r)
    _, _, means = np_get_filter_banks(center_coords_dec = [1,1], sigma_dec = 0, stride_dec = -3, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'r')
    _, _, means = np_get_filter_banks(center_coords_dec = [1,1], sigma_dec = 0, stride_dec = -2, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'b')
    _, _, means = np_get_filter_banks(center_coords_dec = [1,1], sigma_dec = 0, stride_dec = -1, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'g')
    _, _, means = np_get_filter_banks(center_coords_dec = [1,1], sigma_dec = 0, stride_dec = -0, N = 5)
    scat = plt.scatter([m[0] for m in means], [m[1] for m in means], color = 'y')
    plt.show(block=False)

def main():
        
    # With some inspiration from Eric Jang: https://blog.evjang.com/2016/06/understanding-and-implementing.html
    attention = True
    variational = True
    num_time_steps = 7
    num_units_enc = 45
    num_units_dec = 45
    num_latents = 10
    N_read = 4
    N_write = 4
    A = 28
    B = 28

    batch_size = 80

    DO_SHARE = False

    # Recorded variables

    canvas = [None] * num_time_steps
    center_coords_decoded = [None] * num_time_steps
    sigma_decoded = [None] * num_time_steps
    stride_decoded = [None] * num_time_steps
    gamma_decoded = [None] * num_time_steps
    center_coords_decoded_write = [None] * num_time_steps
    sigma_decoded_write = [None] * num_time_steps
    stride_decoded_write = [None] * num_time_steps
    gamma_decoded_write = [None] * num_time_steps
    lms = [None] * num_time_steps
    lss = [None] * num_time_steps

    # Setting up the computational graph

    x = tf.placeholder(shape=[batch_size, A*B],dtype=tf.float32, name = 'x')
    canvas_previous = tf.zeros(shape=[batch_size, A*B])

    encoder_cell = tf.contrib.rnn.LSTMBlockCell(num_units = num_units_enc)
    encoder_state_previous = encoder_cell.zero_state(batch_size, tf.float32)
    def encode_step(input_data, network_state):
        with tf.variable_scope("encoder",reuse=DO_SHARE):
            return encoder_cell(inputs = input_data, state = network_state) # This does one cycle of the RNN
            # This uses the __call__ method

    decoder_cell = tf.contrib.rnn.LSTMBlockCell(num_units = num_units_dec)
    decoder_state_previous = decoder_cell.zero_state(batch_size, tf.float32)
    def decode_step(latents, network_state):
        with tf.variable_scope("decoder",reuse=DO_SHARE):
            return decoder_cell(inputs = latents, state = network_state) # This does one cycle of the RNN
            # This uses the __call__ method

    for t in range(num_time_steps): # Replace with tf.while_loop
        print ("Setting up graph for time-step %i" %t)

        xhat = x-tf.sigmoid(canvas_previous)

        if attention:
            with tf.variable_scope("read_operation",reuse=DO_SHARE):
                read_params = tf.gather(tf.contrib.layers.fully_connected(inputs = decoder_state_previous.h, num_outputs = 5, activation_fn = None), 0)

            center_coords_decoded[t] = [tf.tanh(read_params[0]),tf.tanh(read_params[1])]
            sigma_decoded[t], stride_decoded[t], gamma_decoded[t] = read_params[2], -1*tf.exp(read_params[3]), read_params[4] 
            filter_bank_X, filter_bank_Y = tf_get_filter_banks(center_coords_decoded[t], sigma_decoded[t], stride_decoded[t], N_read)

            read_result = [tf.exp(gamma_decoded[t])*tf.matmul(tf.matmul([filter_bank_Y for i in range (batch_size)], tf.reshape(x, [batch_size,A,B])), [tf.transpose(filter_bank_X) for i in range (batch_size)]), tf.exp(gamma_decoded[t])*tf.matmul(tf.matmul([filter_bank_Y  for i in range (batch_size)], tf.reshape(xhat, [batch_size,A,B])), [tf.transpose(filter_bank_X)  for i in range (batch_size)])] 
            # Some annoying list stuff with the filter matrices in here because it wasn't broadcasting properly and batch_matmul doesn't seem to exist anymore

            # Update the encoder state
            encoder_state = encode_step(input_data = tf.concat([tf.reshape(read_result, [batch_size,N_read*N_read + N_read*N_read]), tf.reshape(decoder_state_previous.h, [batch_size,num_units_dec])], axis=-1), network_state = encoder_state_previous)
            # Note: encoder_state is the output of the LSTMCell's call method -- first output, then a LSTM state tuple
        else:
            encoder_state = encode_step(input_data = tf.concat([x,xhat], axis = -1), network_state = encoder_state_previous)

        # Sample the latents
        with tf.variable_scope("latents",reuse=DO_SHARE):
            latent_means = tf.contrib.layers.fully_connected(encoder_state[0], num_latents, activation_fn = None)
            latent_stds = tf.exp(tf.contrib.layers.fully_connected(encoder_state[0], num_latents, activation_fn = None))

        samples_without_mean = tf.random_normal([num_latents], mean=0, stddev=1, dtype=tf.float32)  
        sampled_latents = latent_means + (latent_stds * samples_without_mean)

        lms[t] = latent_means
        lss[t] = latent_stds

        # Update the decoder state
        decoder_state = decode_step(latents = sampled_latents, network_state = decoder_state_previous)

        # Write
        if attention:
            with tf.variable_scope("write_operation_attn",reuse=DO_SHARE):
                writing_instruction =  tf.reshape(tf.contrib.layers.fully_connected(decoder_state[0], N_write*N_write, activation_fn = None), [batch_size, N_write,N_write])
                write_params = tf.gather(tf.contrib.layers.fully_connected(inputs = decoder_state[0], num_outputs = 5, activation_fn = None), 0)

            center_coords_decoded_write[t] = [tf.tanh(write_params[0]),tf.tanh(write_params[1])]
            sigma_decoded_write[t], stride_decoded_write[t], gamma_decoded_write[t] = write_params[2], -1*tf.exp(write_params[3]), write_params[4]
            filter_bank_X_write, filter_bank_Y_write = tf_get_filter_banks(center_coords_decoded_write[t], sigma_decoded_write[t], stride_decoded_write[t], N_write)

            writing_value = tf.exp(-1*gamma_decoded_write[t])*tf.matmul(tf.matmul([tf.transpose(filter_bank_Y_write) for i in range(batch_size)], writing_instruction), [filter_bank_X_write for i in range(batch_size)])
        else:
            with tf.variable_scope("write_operation_noattn",reuse=DO_SHARE):
                writing_instruction = tf.contrib.layers.fully_connected(decoder_state[0], A*B, activation_fn = None)
            writing_value = writing_instruction

        canvas[t] =  canvas_previous + tf.reshape(writing_value, [batch_size,A*B])

        # Cycle
        encoder_state_previous = encoder_state[1]
        decoder_state_previous = decoder_state[1]
        canvas_previous = canvas[t]

        # To turn this on after the first cycle
        DO_SHARE = True

    canvas = tf.sigmoid(canvas)

    # Setting up loss function
    print("Setting up loss function...")
    epsilon = 0.00000001 # A trick like this was used by Eric Jang for numerical stability, although there with cross-entropy loss -- here it avoids Inf in the log as well
    o = canvas[-1]
    loss_reconstruction = tf.reduce_mean(tf.reduce_sum(-(x*tf.log(o+epsilon) + (1.0-x)*tf.log(1.0-o+epsilon)))) # Cross entropy loss: ~probability of the data
    # I think the reason for this is similar to here: https://stats.stackexchange.com/questions/242907/why-use-binary-cross-entropy-for-generator-in-adversarial-networks/242927?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    loss_latent = tf.reduce_mean(tf.reduce_sum([0.5 * tf.reduce_sum(tf.square(lms[t]) + tf.square(lss[t]) - tf.log(tf.square(lms)+epsilon) - 1) for t in range(num_time_steps)]))
    if variational:
        relative_weight = 1.0 # Between the reconstruction versus latent KL loss
    else:
        relative_weight = 0.0
    loss = loss_reconstruction + relative_weight * loss_latent # this includes a manual scaling parameter
    
    # Setting up the optimizer (we'll have to move this above)
    print("Setting up optimizer...")
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads=optimizer.compute_gradients(loss)
    # train_op=optimizer.minimize(loss)
    for i,(g,v) in enumerate(grads):
        if g is not None:
            grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)
    
    # Sanity check on variable shapes
    print("All variables...")
    for v in tf.all_variables():
        print("\t%s : %s" % (v.name,v.get_shape()))
        
    print("Setting up session...")
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    print("Running...")

    imgs = []
    for b in range(batch_size):
        imgs.append(mnist.train.images[32100+b][:])

    l, cv, lmsout, lssout, ccd, sig, stri, ccdw, sigw, striw = sess.run([loss, canvas, lms, lss, center_coords_decoded, sigma_decoded, stride_decoded, center_coords_decoded_write, sigma_decoded_write, stride_decoded_write], feed_dict = {x:imgs})
    
    print("Untrained model...")
    f, axarr = plt.subplots(1,len(cv))
    for t in range(len(cv)):
        c = cv[t][-1] # Last batch of each time step
        c_img = MNIST_row_as_image(c.tolist())
        axarr[t].set_axis_off()
        axarr[t].imshow(c_img, cmap = cm.Greys_r)
        filter_bank_X, filter_bank_Y, means = np_get_filter_banks(center_coords_dec = ccd[t], sigma_dec = sig[t], stride_dec = stri[t], N = N_read)
        filter_bank_Xw, filter_bank_Yw, meansw = np_get_filter_banks(center_coords_dec = ccdw[t], sigma_dec = sigw[t], stride_dec = striw[t], N = N_write)
        if attention:
            scat = axarr[t].scatter([m[0] for m in means], [m[1] for m in means], color = 'r')
            scat2 = axarr[t].scatter([m[0] for m in meansw], [m[1] for m in meansw], color = 'g')
        ax = plt.gca()
        ax.set_axis_off()    
        
    plt.show(block=False)
        
    print("Training...")
    print("Attention: %r" % attention)
    losses_log = []
    latent_losses_log = []

    num_batches = 50000

    for i in range(num_batches):
        print("Examples %i" % i)
        imgs, ys = mnist.train.next_batch(batch_size)
        l, t_op, ll, cv, lmsout, lssout, ccd, sig, stri, ccdw, sigw, striw = sess.run([loss,  train_op, loss_latent, canvas, lms, lss, center_coords_decoded, sigma_decoded, stride_decoded, center_coords_decoded_write, sigma_decoded_write, stride_decoded_write], feed_dict = {x:imgs})
        if i % 1000 == 0: # Do the plotting on every 1000 batches
            f, axarr = plt.subplots(1,len(cv))
            for t in range(len(cv)):
                c = cv[t][-1] # Last batch of each time step
                c_img = MNIST_row_as_image(c.tolist())
                axarr[t].set_axis_off()
                axarr[t].imshow(c_img, cmap = cm.Greys_r)
                filter_bank_X, filter_bank_Y, means = np_get_filter_banks(center_coords_dec = ccd[t], sigma_dec = sig[t], stride_dec = stri[t], N = N_read)
                filter_bank_Xw, filter_bank_Yw, meansw = np_get_filter_banks(center_coords_dec = ccdw[t], sigma_dec = sigw[t], stride_dec = striw[t], N = N_write)
                if attention:
                    scat = axarr[t].scatter([m[0] for m in means], [m[1] for m in means], color = 'r')
                    scat2 = axarr[t].scatter([m[0] for m in meansw], [m[1] for m in meansw], color = 'g')
                    
            plt.show(block=False)

        losses_log.append(l)
        latent_losses_log.append(ll)
        print("\tLoss %f" % l)
        
    plt.figure()
    plt.title("Training loss")

    conv_size = 1
    def smooth(ser):
        return np.convolve(ser, np.ones((conv_size,))/conv_size, mode='valid')

    plt.plot(smooth(losses_log), label = "loss (reconstruction + latent)")
    plt.legend()
    plt.figure()
    plt.plot(smooth(latent_losses_log), label = "latent loss")
    plt.legend()
    plt.figure()
    plt.plot(smooth([losses_log[k] - relative_weight*latent_losses_log[k] for k in range(len(losses_log))][:]), label = "reconstruction loss")
    plt.legend()

    plt.show(block=False)
    
    imgs = []
    for b in range(batch_size):
        imgs.append(mnist.train.images[32100+b][:])

    print("Trained model...")
    l, cv, lmsout, lssout, ccd, sig, stri, ccdw, sigw, striw = sess.run([loss, canvas, lms, lss, center_coords_decoded, sigma_decoded, stride_decoded, center_coords_decoded_write, sigma_decoded_write, stride_decoded_write], feed_dict = {x:imgs})
    
    f, axarr = plt.subplots(1,num_time_steps)
    for t in range(len(cv)):
        c = cv[t][-1] # Last batch of each time step
        c_img = MNIST_row_as_image(c.tolist())
        axarr[t].set_axis_off()
        axarr[t].imshow(c_img, cmap = cm.Greys_r)
        filter_bank_X, filter_bank_Y, means = np_get_filter_banks(center_coords_dec = ccd[t], sigma_dec = sig[t], stride_dec = stri[t], N = N_read)
        filter_bank_Xw, filter_bank_Yw, meansw = np_get_filter_banks(center_coords_dec = ccdw[t], sigma_dec = sigw[t], stride_dec = striw[t], N = N_write)
        if attention:
            scat = axarr[t].scatter([m[0] for m in means], [m[1] for m in means], color = 'r')
            scat2 = axarr[t].scatter([m[0] for m in meansw], [m[1] for m in meansw], color = 'g')
            
    plt.show(block=False)
    
    plt.show()
        
if __name__ == '__main__':
    main()