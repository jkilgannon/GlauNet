print(model.count_params())
print("++++++++++++++")
print(model.summary())
print("++++++++++++++")
os.system('free -m')
print("++++++++++++++")
os.system('vmstat -s')
print("++++++++++++++")


"""
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""

"""
EARLYSTOP = EarlyStopping(patience=50, 
                          monitor='val_categorical_accuracy', 
                          restore_best_weights=True)
EARLYSTOP = EarlyStopping(patience=50, 
                          monitor='binary_crossentropy', 
                          restore_best_weights=True)
"""

EARLYSTOP = EarlyStopping(patience=50, 
                          monitor='cosine_similarity', 
                          restore_best_weights=True)

"""
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor='val_categorical_accuracy', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor='binary_crossentropy', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)
"""

# Save off the very best model we can find; avoids overfitting.
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor='cosine_similarity', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)

"""
history = model.fit_generator(batchmaker_train(),
                    steps_per_epoch=num_fl // batchsize,
                    shuffle=True, 
                    epochs=500,
                    validation_data=batchmaker_test(),
                    validation_steps=num_fl_val // batchsize,
                    callbacks=[EARLYSTOP, CHKPT])
"""

history = model.fit_generator(batchmaker_train(),
                    steps_per_epoch=num_fl // batchsize,
                    shuffle=True, 
                    epochs=500,
                    validation_data=batchmaker_test(),
                    validation_steps=num_fl_val // batchsize)

model.save_weights(out_path + 'last_weights.h5') 
