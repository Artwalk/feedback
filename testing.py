from textgenrnn import textgenrnn

print('👍')
textgenrnn('weights/👍.hdf5').generate_samples(prefix="He")

print('👎')
textgenrnn('weights/👎.hdf5').generate_samples(prefix="He")
