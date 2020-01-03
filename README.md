# Is a GPU really more powerfull than a CPU? 

Copied from my answer at [Stackoverflow](https://stackoverflow.com/questions/59567162/is-a-gpu-really-more-powerfull-than-a-cpu) on 2nd, January of 2020th. [closed by CUDA inquisitors]

I have done a test using CUDA with my laptop. Using the CPU (i7 6700HQ) is at least 20 times more powerful than the 65,000 processors of the GEFORCE 940M inside. While with CUDA it takes about 20 seconds to find 50,000 items over a set of 1 million, the CPU does it in less than 1 second.

What I´m doing wrong? This is the CUDA code (c++):

    __global__ void FindFirst(const int items_count, SomeBasicType* data, long target, int* pFound)
	{						
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < items_count && *pFound < 0; i += blockDim.x * gridDim.x)
			{				
				if (data[i].Value == target) *pFound = i;
			}
	}

*** EDIT 1: *** 

After a few hours checking what I was doing wrong, the conclusion is that CUDA is limited to work isolated from CPU and it is only efficient for very simple algorithms. Implementing a dictionary in C# is exponentially more efficient than any kernel you can write in CUDA. 

I followed the code of Marc Gravell published [here][1]; adding a new kernel for search and some code to test in main sub. You can get it from my GitHub [CUDA_kernel_0][2].


  [1]: https://github.com/mgravell/SimpleCUDAExample/tree/master/Demo
  [2]: https://github.com/IzarUrdin/CUDA_kernel_0
