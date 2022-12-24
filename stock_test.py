from stock import *
from sklearn.metrics import f1_score


text_proc = TextProc('stockerbot-export.csv')

train_x, val_x, train_y, val_y = train_test_split(text_proc.sent,
                        text_proc.labels, test_size=0.3)

class_weights = compute_class_weights(text_proc.df['verified'])
print(class_weights)

model = define_hud(0.3, 16, 16, 1, 'sigmoid')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
num_epochs = 100
history = model.fit(train_x, train_y, epochs=num_epochs, validation_data=(val_x, val_y),
                    class_weight=class_weights, callbacks=callback)

graph_plots(history, "accuracy")
graph_plots(history, "loss")

def test_model():
    loss, val_acc = model.evaluate(val_x, val_y)
    assert val_acc > 0.93

def test_stock():
    sent = ["今年は景気が悪く利益がほとんどなかったためボーナスなんてなかった."]
    print(model.predict(sent))
    pred = np.round(model.predict(sent))
    assert pred == 0