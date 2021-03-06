﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Demo
{
    struct ValueIdItem
    {
        // yes, these are public mutable fields; we are explicitly **not**
        // trying to provide abstractions here - we're holding our hands
        // up and saying "you're playing with raw memory, don't screw up"
        public int Id;
        public uint Value;
    }
    unsafe sealed class GPU : IDisposable
    {
        public void Dispose() => Dispose(true);
        ~GPU() { Dispose(false); } // we don't want to see this... this means we failed
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this); // dispose was called correctly
            }
            // release the local buffer - note we want to do this while the CUDA context lives
            if (hostBuffer != default(ValueIdItem*))
            {
                var tmp = new IntPtr(hostBuffer);
                hostBuffer = default(ValueIdItem*);
                try
                {
                    DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(tmp);
                }
                catch (Exception ex) { Debug.WriteLine(ex.Message); }
            }

            if (disposing) // clean up managed resources
            {
                Dispose(ref deviceBuffer);
                Dispose(ref defaultStream);
                Dispose(ref ctx);
            }
        }
        // utility method to dispose and wipe fields
        private void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try { field.Dispose(); } catch (Exception ex) { Debug.WriteLine(ex.Message); }
                field = null;
            }
        }

        public GPU(int deviceId, int _count = -1)
        {
            // note that this initializes a lot of things and binds *to the thread*
            ctx = new CudaContext(deviceId, true);
            if (_count > 0) count = _count;
            var props = ctx.GetDeviceInfo();
            defaultBlockCount = props.MultiProcessorCount * 32;
            defaultThreadsPerBlock = props.MaxThreadsPerBlock;
            warpSize = props.WarpSize;
        }

        private int count, defaultBlockCount, defaultThreadsPerBlock, warpSize;
        private ValueIdItem* hostBuffer;
        CudaDeviceVariable<ValueIdItem> deviceBuffer;
        CudaContext ctx;
        CudaStream defaultStream;
        CudaKernel multiply;
        CudaKernel findFirst;
        CudaKernel parallelFor;
        public void InitializeData(int count)
        {
            if (count < 1) throw new ArgumentOutOfRangeException(nameof(count));
            this.count = count;

            // allocate a buffer at the host (meaning: accessible to the CPU)
            // note: for client-side, we *don't* want to just use an array, 
            // as we want the maximum GPU<===>CPU memory transfer speed,
            // which requires fixed pages allocated by the GPU driver
            IntPtr hostPointer = IntPtr.Zero;
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref hostPointer, count * sizeof(ValueIdItem));
            if (res != CUResult.Success) throw new CudaException(res);
            hostBuffer = (ValueIdItem*)hostPointer;

            // allocate a buffer at the device (meaning: accessible to the GPU)
            deviceBuffer = new CudaDeviceVariable<ValueIdItem>(count);

            // initialize the local data
            for (int i = 0; i < count; i++)
            {
                // we'll just set the key and value to i, so: [{0,0},{1,1},{2,2}...]
                hostBuffer[i].Id = i;
                hostBuffer[i].Value = (uint)i;
            }

            // allocate a stream for async/overlapped operations; note we're only going to use
            // one stream, but in complex code you can use different streams to allow concurrent
            // memory transfer while unrelated kernels execute, etc
            defaultStream = new CudaStream();

            // transfer the local buffer to the server (note that CudaDeviceVariable<T> also exposes
            // various methods to do this, but not this particular set using raw memory and streams)
            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceBuffer.DevicePointer,
                hostPointer, deviceBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success) throw new CudaException(res);

        }

        private static int RoundUp(int value, int blockSize)
        {
            if ((value % blockSize) != 0)
            {   // take away the surplus, and add an entire extra block
                value += blockSize - (value % blockSize);
            }
            return value;
        }

        internal long ParallelFor(long value)
        {
            result.CopyToDevice(new long[] { 0 });

            parallelFor.Run(new object[]
                    { value, result.DevicePointer});

            long[] returned = new long[1];
            result.CopyToHost(returned);
            return returned[0];
        }
        internal long FindFirst(long value)
        {

            result.CopyToDevice(new long[] { -1 });

            // invoke the kernel 
            findFirst.Run(new object[]
                    { deviceBuffer.DevicePointer, value, result.DevicePointer});

            long[] returned = new long[1];
            result.CopyToHost(returned);
            return returned[0];
        }
        internal void MultiplyAsync(int value)
        {
            // invoke the kernel
            multiply.RunAsync(defaultStream.Stream, new object[] {
                // note the signature is (N, data, factor)
                count, deviceBuffer.DevicePointer, value
            });
        }

        int threadsPerBlock, blockCount;
        CudaDeviceVariable<long> result;
        internal void PrepareExecution()
        {
            // configure the dimensions; note, usually this is a lot more dynamic based
            // on input data, but we'll still go through the motions

            if (count <= defaultThreadsPerBlock) // a single block
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, warpSize); // slight caveat here; if you are using "shuffle" operations, you
                                                            // need to use entire "warp"s - otherwise the result is undefined
            }
            else if (count >= defaultThreadsPerBlock * defaultBlockCount)
            {
                // more than enough work to keep us busy; just use that
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = defaultBlockCount;
            }
            else
            {
                // do the math to figure out how many blocks we need
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }

            // we're using 1-D math, but actually CUDA supports blocks and grids that span 3 dimensions
            multiply.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            multiply.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            findFirst.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            findFirst.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            parallelFor.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            parallelFor.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            result = new CudaDeviceVariable<long>(1);

        }
        internal void Synchronize()
        {
            ctx.Synchronize(); // this synchronizes (waits for) **all streams**
            // to synchronize a single stream, use {theStream}.Synchronize();
        }

        internal ManagedCuda.NVRTC.nvrtcResult LoadKernel(out string log)
        {
            string path = "MyKernels.c";
            ManagedCuda.NVRTC.nvrtcResult result;
            using (var rtc = new ManagedCuda.NVRTC.CudaRuntimeCompiler(File.ReadAllText(path), Path.GetFileName(path)))
            {
                try
                {
                    rtc.Compile(new string[0]); // see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
                    result = ManagedCuda.NVRTC.nvrtcResult.Success;
                }
                catch (ManagedCuda.NVRTC.NVRTCException ex)
                {
                    result = ex.NVRTCError;
                }
                log = rtc.GetLogAsString();

                if (result == ManagedCuda.NVRTC.nvrtcResult.Success)
                {
                    byte[] ptx = rtc.GetPTX();
                    multiply = ctx.LoadKernelFatBin(ptx, "Multiply"); // hard-coded method name from the CUDA kernel
                    findFirst = ctx.LoadKernelFatBin(ptx, "FindFirst"); // MY method name from the CUDA kernel
                    parallelFor = ctx.LoadKernelFatBin(ptx, "ParallelFor"); // MY 2nd method name from the CUDA kernel
                }
            }
            return result;
        }

        internal void CopyToHost()
        {
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(
                new IntPtr(hostBuffer), deviceBuffer.DevicePointer, deviceBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        public ValueIdItem this[long index]
        {
            get
            {
                // note: whenever possible, try to avoid doing *per call* range checks; this is purely
                // for illustration; you should prefer working with *blocks* of data in one go
                //if (index < 0 || index >= count) throw new IndexOutOfRangeException();
                return hostBuffer[index]; // note: here, the data is copied from localBuffer to the stack
            }
        }


    }
}
