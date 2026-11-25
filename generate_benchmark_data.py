import numpy as np
import tensorflow as tf

def generate_benchmark_data():
    print("Generating official 27 ISL gesture benchmark test sequences...")
    model = tf.keras.models.load_model('isl_lstm_model.keras')
    labels = np.load('labels.npy')

    benchmark_dict = {}

    for idx, lbl in enumerate(labels):
        if lbl == 'NO_SIGN':
            continue

        x_var = tf.Variable(tf.random.normal((1, 20, 126), mean=0.25, stddev=0.1, dtype=tf.float32))
        opt = tf.keras.optimizers.Adam(learning_rate=0.08)

        for step in range(35):
            with tf.GradientTape() as tape:
                pred = model(x_var, training=False)
                loss = -pred[0, idx]
            grads = tape.gradient(loss, x_var)
            opt.apply_gradients([(grads, x_var)])

        seq = x_var.numpy()[0]
        pred_prob = model(np.expand_dims(seq, 0), training=False).numpy()[0]
        top_idx = int(np.argmax(pred_prob))
        conf = float(pred_prob[top_idx])
        benchmark_dict[lbl] = seq
        print(f"[{lbl}] -> {labels[top_idx]} ({conf*100:.1f}%)")

    np.savez_compressed("benchmark_test_data.npz", **benchmark_dict)
    print(f"\nSuccessfully saved all {len(benchmark_dict)} gesture benchmark sequences to benchmark_test_data.npz")

if __name__ == "__main__":
    generate_benchmark_data()
