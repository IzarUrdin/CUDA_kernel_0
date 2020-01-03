using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace Demo
{
    class Program
    {
        static int Main()
        {
            //Application.Run(new FormX());
            
            var scr = Console.ReadLine();
            while (scr.ToLower() != "do")
            {
                scr = Console.ReadLine();
            }

            try
            {
                int deviceCount = CudaContext.GetDeviceCount();
                if (deviceCount == 0)
                {
                    Console.Error.WriteLine("No CUDA devices detected. Sad face.");
                    return -1;
                }
                Console.WriteLine($"{deviceCount} CUDA devices detected (first will be used)");
                for (int i = 0; i < deviceCount; i++)
                {
                    Console.WriteLine($"{i}: {CudaContext.GetDeviceName(i)}");
                }
                const int Count = 1024*46000;
                using (var myGPU = new GPU(deviceId: 0))
                {
                    Console.WriteLine("Initializing kernel...");
                    string log;
                    var compileResult = myGPU.LoadKernel(out log);
                    if(compileResult != ManagedCuda.NVRTC.nvrtcResult.Success)
                    {
                        Console.Error.WriteLine(compileResult);
                        Console.Error.WriteLine(log);
                        Console.ReadLine();
                        return -1;
                    }
                    Console.WriteLine(log);

                    Console.WriteLine("Initializing data...");
                    myGPU.InitializeData(Count);
                    myGPU.PrepareExecution();
                    Console.WriteLine("Running kernel...");
                    for (int i = 0; i < 8; i++)
                    {
                        myGPU.MultiplyAsync(2);
                    }

                    Console.WriteLine("Copying data back...");
                    myGPU.CopyToHost(); // note: usually you try to minimize how much you need to
                    // fetch from the device, as that can be a bottleneck; you should prefer fetching
                    // minimal aggregate data (counts, etc), or the required pages of data; fetching
                    // *all* the data works, but should be avoided when possible.

                    Console.WriteLine("Waiting for completion...");
                    myGPU.Synchronize();
                    var tests = 10000;
                    var random = new Random(123456);
                    ValueIdItem[] randoms = new ValueIdItem[Count];
                    Dictionary<long, long> DiRecs = new Dictionary<long, long>();

                    
                    try
                    {
                        for (int i = 0; i < Count; i++)
                        {
                            DiRecs.Add(i, myGPU[i].Value);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                    }

                    for (int i = 0; i < tests; i++)
                    {
                        randoms[i] = myGPU[random.Next(Count)];
                    }
                    Console.WriteLine("all done; testing to find " + tests + " items");
                    var startTime = DateTime.Now;
                    for (int i = 0; i < tests; i++)
                    {                        
                        //Console.WriteLine($"{i}: {nameof(record.Id)}={record.Id}, {nameof(record.Value)}={record.Value}");
                        var result = myGPU.FindFirst(myGPU[1].Value);
                       
                        //if (result < 0)
                        //{
                        //    Console.WriteLine("Not found");
                        //}
                        //else
                        //{ Console.WriteLine("Found at " + result);
                        //}
                    }                   

                    var endTime = DateTime.Now;
                    Console.WriteLine("GPU test done:");
                    Console.WriteLine(tests / ((endTime-startTime).TotalSeconds+1) + " matchins per second !");

                    startTime = DateTime.Now;
                    for (int i = 0; i < tests; i++)
                    {
                        var result = DiRecs.ContainsKey(randoms[i].Id);
                    }
                    endTime = DateTime.Now;

                    Console.WriteLine("CPU test done:");
                    Console.WriteLine(tests / ((endTime - startTime).TotalSeconds+1) + " matchins per second !");

                    Console.WriteLine("Cleaning up...");                    
                }
                
                Console.WriteLine("All done; have a nice day");
                
            } catch(Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
              
            }
            Console.ReadLine();
            return 0;
        }
    }
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
            if(disposing)
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
                } catch(Exception ex) { Debug.WriteLine(ex.Message); }
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
            if(field != null)
            {
                try { field.Dispose(); } catch (Exception ex) { Debug.WriteLine(ex.Message); }
                field = null;
            }
        }

        public GPU(int deviceId)
        {
            // note that this initializes a lot of things and binds *to the thread*
            ctx = new CudaContext(deviceId, true);

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

        internal long FindFirst(long value)
        {
            
            result.CopyToDevice(new int[]{ -1});

            // invoke the kernel 
            findFirst.Run(new object[]
                    { count, deviceBuffer.DevicePointer, value,result.DevicePointer});
            //findFirst.RunAsync
            //    (
            //      defaultStream.Stream, 
            //      new object[] 
            //        { count, deviceBuffer.DevicePointer, value,result.DevicePointer}
            //    );

            //int[] returned = new int[1];

            //result.CopyToHost(returned);
            //return returned[0];
            return 1;
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
        CudaDeviceVariable<int> result;
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
            findFirst.GridDimensions = new ManagedCuda.VectorTypes.dim3(count/threadsPerBlock, 1, 1);

            result = new CudaDeviceVariable<int>(1);

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
                } catch(ManagedCuda.NVRTC.NVRTCException ex)
                {
                    result = ex.NVRTCError;
                }
                log = rtc.GetLogAsString();

                if (result == ManagedCuda.NVRTC.nvrtcResult.Success)
                {
                    byte[] ptx = rtc.GetPTX();
                    multiply = ctx.LoadKernelFatBin(ptx, "Multiply"); // hard-coded method name from the CUDA kernel
                    findFirst = ctx.LoadKernelFatBin(ptx, "FindFirst"); // MY method name from the CUDA kernel
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
