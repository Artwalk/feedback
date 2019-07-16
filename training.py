from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('datasets/👍.txt', num_epochs=1)
textgen.save('weights/👍.hdf5')

textgen.train_from_file('datasets/👎.txt', num_epochs=1)
textgen.save('weights/👎.hdf5')
