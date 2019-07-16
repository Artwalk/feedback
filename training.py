from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('datasets/👍.txt',
                        rnn_layers=5,
                        num_epochs=25, rnn_bidirectional=True)
textgen.save('weights/👍.hdf5')

# textgen.train_from_file('datasets/👎.txt', num_epochs=1)
# textgen.save('weights/👎.hdf5')
