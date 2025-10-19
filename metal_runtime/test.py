import objc
from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
from Foundation import NSData
import numpy as np
from ctypes import c_void_p, cast, memmove

# --- Step 1: Create the buffer (this part works) ---
device = MTLCreateSystemDefaultDevice()
data = np.arange(10, dtype=np.float32)
data_bytes = data.tobytes()
ns_data = NSData.dataWithBytes_length_(data_bytes, len(data_bytes))

buffer = device.newBufferWithBytes_length_options_(
    ns_data,
    len(data_bytes),
    MTLResourceStorageModeShared
)

print(f"Created a buffer with length: {buffer.length()}")

# --- Step 2: The Hack - Manual Memory Read ---
# We can't use np.frombuffer or ctypes.cast directly on buffer.contents().
# But we CAN use memmove to copy the data from the buffer's memory
# into a Python-controlled memory block.

# 1. Get the raw pointer to the buffer's memory.
#    We use objc.pyobjc_id to get a handle that PyObjC can work with.
buffer_ptr_handle = objc.pyobjc_id(buffer.contents())

# 2. Create a Python bytes object of the correct size to act as a destination.
#    This is a "dumb" container that we can copy data into.
readback_bytes = bytearray(buffer.length())

# 3. Use memmove to copy the data FROM the buffer TO our bytes object.
#    This is the core of the hack. We're telling C to copy memory.
memmove(readback_bytes, buffer_ptr_handle, buffer.length())

# 4. Now that we have the data in a standard Python bytes object,
#    NumPy can finally understand it.
readback_array = np.frombuffer(readback_bytes, dtype=np.float32)

print("Original data:", data)
print("Read-back data:", readback_array)
