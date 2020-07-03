from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss = 'binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['acc']
)
