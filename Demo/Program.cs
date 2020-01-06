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
            Console.Write("Press enter to test your GPU");
            
            var scr = Console.ReadLine();
                       
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

                bool GoOn = true;

                while (GoOn)
                {
                    //for (int z = 100000; z < 100000*3000; z = z*2)
                    //{
                    long z = 50000000;
                    for (int a = 2; a < 3; a++)
                    {
                        Console.WriteLine($"GridDim:{a} n:{z / 1000000} MM");
                        using (var myGPU = new GPU(deviceId: 0, _count: 1024 * a))
                        {
                            // Console.WriteLine("Initializing kernel...");
                            string log;
                            var compileResult = myGPU.LoadKernel(out log);
                            if (compileResult != ManagedCuda.NVRTC.nvrtcResult.Success)
                            {
                                Console.Error.WriteLine(compileResult);
                                Console.Error.WriteLine(log);
                                Console.ReadLine();
                                return -1;
                            }
                            //Console.WriteLine(log);

                            //Tests.Test_0(Count, myGPU);

                            Tests.Test_1(myGPU, 1, z);

                            //Tests.Test_2(z, myGPU);
                        }

                        Console.WriteLine("Cleaning up...");
                    }
                    //     Console.ReadKey();
                    //}

                    Console.WriteLine("All done; ESC to exit any key to repeat :)");
                    var k = Console.ReadKey();
                    GoOn = k.Key.ToString() != Keys.Escape.ToString();
                }
            } catch(Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
              
            }
           
            return 0;
        }

    }
    
}
