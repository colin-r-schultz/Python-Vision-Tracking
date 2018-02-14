import struct
floats = [0.5, 0.25, 0.1]
s = struct.pack('>3f', *floats)
f = open('bytes','wb')
f.write(s)
f.close()
