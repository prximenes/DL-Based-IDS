import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# Pruning parameters
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 256
epochs = 6
validation_split = 0.1

num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.90,
        begin_step=0,
        end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
]

history_pruning = model_for_pruning.fit(
    x_train, y_train,
    epochs=epochs,
    validation_split=validation_split,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)
