using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Demo
{
    class Tests
    {
        public static void Test_1(GPU myGPU, long tests, int n = 90000000)
        {
            
            myGPU.PrepareExecution();
            long result = 0;
            var startTime = DateTime.Now;
            for (int i = 0; i < tests; i++)
            {
                result = myGPU.ParallelFor(n); // each thread in each block iterates n 
            }
            myGPU.Synchronize();
            var endTime = DateTime.Now;
            Console.WriteLine("Operations : " + result + " in " + (result/n) + " threads in " + (endTime - startTime).TotalMilliseconds + " ms.");
            Console.WriteLine(ToInt(tests * result / ((endTime - startTime).TotalMilliseconds * 1000)) + " MM operations per second in GPU");

            long max = 0;
            startTime = DateTime.Now;
            for (long t = 0; t < tests; t++)
            {
                for (long i = 0; i < result + 1; i++)
                {
                    max = max + 1; //total number of executions
                }
            }
            endTime = DateTime.Now;
            Console.WriteLine(ToInt(tests * max / ((endTime - startTime).TotalMilliseconds * 1000)) + " MM operations per second in CPU");

            // Console.ReadKey();

        }

        private static long ToInt(double v)
        {
            return (long)v;
        }

        public static void Test_0(int Count, GPU myGPU)
        {
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

            var tests = 100000;
            var random = new Random(123456);
            ValueIdItem[] randoms = new ValueIdItem[tests];
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
                long z = random.Next(Count - 1);
                try
                {
                    randoms[i] = myGPU[z];
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }

            Console.WriteLine("all done; testing to find " + tests + " items");
            var startTime = DateTime.Now;
            for (int i = 0; i < tests; i++)
            {
                //Console.WriteLine($"{i}: {nameof(record.Id)}={record.Id}, {nameof(record.Value)}={record.Value}");
                var result = myGPU.FindFirst(myGPU[i].Value);

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
            Console.WriteLine(tests / ((endTime - startTime).TotalSeconds) + " matchins per second !");

            startTime = DateTime.Now;
            for (int i = 0; i < tests; i++)
            {
                var result = DiRecs[randoms[i].Id];
            }
            endTime = DateTime.Now;

            Console.WriteLine("CPU test done:");
            Console.WriteLine(tests / ((endTime - startTime).TotalSeconds) + " matchins per second !");
        }
    }
}
