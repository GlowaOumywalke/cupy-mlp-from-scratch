import cupy as cp

print("Devices:", cp.cuda.runtime.getDeviceCount())

device = cp.cuda.Device(0)
device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
print("Name:", device_name)

x = cp.arange(10)
print("Array:", x)
