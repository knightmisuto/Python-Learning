import numpy as np
import tensorflow as tf

Buttons = np.load("Data/Data/Buttons.npy")
Skills = np.load("Data/Data/Skills.npy")

Buttons_list = ["down_normal", "down_reverse", "end_normal", "end_reverse",
           "home_normal", "home_reverse", "left_normal", "left_reverse",
           "pgdn_normal", "pgdn_reverse", "pgup_normal", "pgup_reverse",
           "right_normal", "right_reverse", "up_normal", "up_reverse"]

Skills = Skills/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 30, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(Buttons_list))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(Skills, Buttons, validation_split=0.2, epochs=10)